
import paddle
from paddle import Tensor

paddle.enable_compat(scope={"triton"})
paddle.set_printoptions(linewidth=200)

import triton
import triton.language as tl


# The radix key contains only the order-preserving float32 bits. Equal scores
# intentionally share a key; gather order provides deterministic tie handling.


@triton.jit
def _encode_key(x):
    """float32 score -> sortable int64 key containing an unsigned 32-bit value."""
    ui = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    neg = (ui >> 31) != 0
    # negative -> flip all bits, non-negative -> flip only the sign bit
    return tl.where(neg, ui ^ 0xFFFFFFFF, ui ^ 0x80000000)


@triton.jit
def _radix_select_gather_kernel(
    scores_ptr,     # [q_len, k_len] float32
    counts_ptr,     # [q_len] int32
    out_idx_ptr,    # [q_len, top_k] int32
    out_score_ptr,  # [q_len, top_k] float32
    k_len,
    top_k,
    stride_sq,
    stride_sk,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    N_BINS: tl.constexpr,
    N_PASSES: tl.constexpr,
):
    """MSD radix-select the top_k-th score, then gather up to K qualifiers.

    Equal scores share a key. If the split score is tied, the deterministic
    block scan and prefix sums keep the first qualifying elements."""
    row = tl.program_id(0)
    cnt = tl.load(counts_ptr + row).to(tl.int32)
    cnt = tl.maximum(0, tl.minimum(cnt, k_len))
    bins = tl.arange(0, N_BINS)

    # ---- radix select: resolve the split key T bit-window by bit-window ----
    prefix = tl.zeros((), tl.int64)   # high bits fixed so far
    k_rem = top_k                     # rank still to locate within `prefix`
    for p in tl.static_range(N_PASSES):
        shift = (N_PASSES - 1 - p) * RADIX_BITS
        hp = shift + RADIX_BITS
        hist = tl.zeros((N_BINS,), tl.int32)
        for start in range(0, cnt, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            in_range = offs < cnt
            x = tl.load(
                scores_ptr + row * stride_sq + offs * stride_sk,
                mask=in_range,
                other=float("-inf"),
            )
            e = _encode_key(x)
            match = ((e >> hp) == (prefix >> hp)) & in_range
            dig = ((e >> shift) & (N_BINS - 1)).to(tl.int32)
            # tl.histogram does the block-wide bin count in one optimized op;
            # `match` drops elements that don't share the fixed prefix.
            hist += tl.histogram(dig, N_BINS, mask=match)
        # walk buckets high -> low, pick the one holding the k_rem-th element
        ge = tl.flip(tl.cumsum(tl.flip(hist, 0), 0), 0)  # inclusive suffix sum
        gt = ge - hist                                   # strictly-higher buckets
        sel = (gt < k_rem) & (ge >= k_rem)               # exactly one bucket
        chosen = tl.sum(tl.where(sel, bins, 0))
        above = tl.sum(tl.where(sel, gt, 0))
        prefix = prefix | (chosen.to(tl.int64) << shift)
        k_rem = k_rem - above

    # ---- gather: compact the first top_k keys >= T directly into output ----
    split = prefix
    count = tl.zeros((), tl.int32)
    for start in range(0, cnt, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        in_range = offs < cnt
        x = tl.load(
            scores_ptr + row * stride_sq + offs * stride_sk,
            mask=in_range,
            other=float("-inf"),
        )
        e = _encode_key(x)
        qual = (e >= split) & in_range
        qi = qual.to(tl.int32)
        pos = count + tl.cumsum(qi, 0) - qi   # exclusive prefix + running count
        store = qual & (pos < top_k)
        tl.store(out_idx_ptr + row * top_k + pos, offs, mask=store)
        tl.store(out_score_ptr + row * top_k + pos, x, mask=store)
        count += tl.sum(qi)

    # Invalid outputs follow the -1 / -inf API contract.
    tail = tl.arange(0, K)
    tail_mask = (tail >= cnt) & (tail < top_k)
    tl.store(out_idx_ptr + row * top_k + tail, -1, mask=tail_mask)
    tl.store(
        out_score_ptr + row * top_k + tail,
        float("-inf"),
        mask=tail_mask,
    )


def indexer_topk_radix(scores: Tensor, counts: Tensor, top_k: int):
    """Deterministic radix-select top-k without output ordering guarantees.

    Finds the top_k split value with a multi-pass radix select, gathers the
    first top_k qualifying elements, and returns them in deterministic gather
    order. Re-reads `scores` several times but does not encode indices into the
    radix key or sort the result."""
    q_len, k_len = scores.shape
    K = triton.next_power_of_2(top_k)

    out_idx = paddle.empty([q_len, top_k], dtype="int32")
    out_score = paddle.empty([q_len, top_k], dtype="float32")

    grid = (q_len,)
    _radix_select_gather_kernel[grid](
        scores, counts, out_idx, out_score,
        k_len, top_k,
        k_len, 1,
        K=K, BLOCK=256,
        RADIX_BITS=6, N_BINS=64, N_PASSES=6,
    )
    return out_idx, out_score


def _numpy_ref(scores_np, counts_np, top_k):
    """Sorted reference used to validate the unordered kernel result."""
    import numpy as np

    q_len, k_len = scores_np.shape
    idx_out = np.full((q_len, top_k), -1, dtype=np.int32)
    score_out = np.full((q_len, top_k), -np.inf, dtype=np.float32)
    for r in range(q_len):
        cnt = int(counts_np[r])
        s = scores_np[r].astype(np.float32).copy()
        s[cnt:] = -np.inf
        # stable sort keeps original order on ties -> smaller index first
        order = np.argsort(-s, kind="stable")[:top_k]
        score_out[r] = s[order]
        take = min(cnt, top_k)
        idx_out[r, :take] = order[:take]
    return idx_out, score_out


if __name__ == "__main__":
    import numpy as np

    paddle.seed(0)
    np.random.seed(0)

    # q dim is independent -> keep it small for a fast test. k_len / top_k large.
    q_len, k_len, top_k = 32768, 8192, 512

    scores = paddle.randn([q_len, k_len], "float32") * 1.0
    counts = paddle.randint(0, k_len + 1, [q_len]).astype("int32")
    # exercise the boundary cases explicitly
    counts[0] = 0            # empty row -> all -1 / -inf
    counts[1] = 1            # fewer valid than top_k
    counts[2] = top_k        # exactly top_k valid
    counts[3] = k_len        # full row

    valid_mask = paddle.arange(k_len, dtype="int32") < counts[:, None]
    scores = paddle.where(valid_mask, scores, paddle.full_like(scores, float("-inf")))

    paddle.topk(scores, top_k, axis=-1)

    idx, sc = indexer_topk_radix(scores, counts, top_k)
    paddle.device.synchronize()
    exit()

    idx_np, sc_np = idx.numpy(), sc.numpy()

    scores_np = scores.numpy()
    counts_np = counts.numpy()
    _, ref_sc = _numpy_ref(scores_np, counts_np, top_k)

    sc_ok = np.array_equal(
        np.sort(sc_np, axis=1),
        np.sort(ref_sc, axis=1),
    )
    idx_ok = True
    for r in range(q_len):
        take = min(int(counts_np[r]), top_k)
        valid_idx = idx_np[r, :take]
        idx_ok &= (
            np.all((valid_idx >= 0) & (valid_idx < int(counts_np[r])))
            and len(np.unique(valid_idx)) == take
            and np.array_equal(sc_np[r, :take], scores_np[r, valid_idx])
            and np.all(idx_np[r, take:] == -1)
        )
    print(f"[correctness] index/score pairs valid: {idx_ok}")
    print(f"[correctness] top-k score multisets match ref: {sc_ok}")

    # determinism: identical bytes across independent runs
    idx2, sc2 = indexer_topk_radix(scores, counts, top_k)
    det_ok = np.array_equal(idx2.numpy(), idx_np) and np.array_equal(
        sc2.numpy(), sc_np
    )
    print(f"[determinism] two runs identical: {det_ok}")

    # Exercise a top-k boundary that cuts through equal scores.
    tie_scores = paddle.to_tensor(
        np.tile(np.arange(8, dtype=np.float32) % 3, (4, 1)), "float32"
    )  # values repeat -> forces ties
    tie_counts = paddle.to_tensor([8, 8, 8, 8], "int32")
    tie_top_k = 5
    t_idx, t_sc = indexer_topk_radix(tie_scores, tie_counts, tie_top_k)
    t_idx2, t_sc2 = indexer_topk_radix(tie_scores, tie_counts, tie_top_k)
    _, r_sc = _numpy_ref(tie_scores.numpy(), tie_counts.numpy(), tie_top_k)
    tie_ok = np.array_equal(
        np.sort(t_sc.numpy(), axis=1), np.sort(r_sc, axis=1)
    ) and all(len(np.unique(row)) == tie_top_k for row in t_idx.numpy())
    tie_ok &= np.array_equal(t_idx.numpy(), t_idx2.numpy())
    tie_ok &= np.array_equal(t_sc.numpy(), t_sc2.numpy())
    print(f"[ties]       selected score multisets deterministic: {tie_ok}")

    assert idx_ok and sc_ok and det_ok and tie_ok, "topk kernel verification FAILED"
    print("ALL CHECKS PASSED")
