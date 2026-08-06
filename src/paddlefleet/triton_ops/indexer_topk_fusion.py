
import paddle
from paddle import Tensor

paddle.enable_compat(scope={"triton"})
paddle.set_printoptions(linewidth=200)

import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Deterministic composite key
#
# We pack each element into a single int64 key so the ordering is a *total*
# order with no ties -> the result is bit-exact deterministic regardless of
# reduction / thread order (unlike cudnn topk which uses nondeterministic
# atomics).
#
#   key64 = (order_preserving_u32(score) << 20) | (0xFFFFF - index)
#
# - The high 32 bits are an order-preserving remap of the float32 bits, so
#   integer ">" on them matches float ">" on the score (works for +-inf too).
# - The low 20 bits store (MAX - index), so when scores tie the *smaller*
#   index gets the larger key -> "equal score picks the earlier index".
# 20 bits supports k_len up to ~1M, well beyond the 32k range here.
# --------------------------------------------------------------------------- #
_LOW_BITS = tl.constexpr(20)
_LOW_MASK = tl.constexpr((1 << 20) - 1)  # 0xFFFFF
# key32 of -inf == 0x007FFFFF; with low bits 0 this is the smallest "real" key,
# so an all -inf buffer never outranks any finite element.
_SENTINEL = tl.constexpr(0x7FFFFF << 20)

# Radix-select config. The composite key uses bits [0, 52) (32 score + 20 index).
# 26 passes of 2 bits cover it (52 >= 52); the top window's high bits are just 0.
# Wider radix => fewer passes => fewer full re-reads of the row, but a wider
# [BLOCK, N_BINS] one-hot buffer per block.
_RADIX_BITS = tl.constexpr(4)
_N_BINS = tl.constexpr(16)     # 2 ** _RADIX_BITS
_N_PASSES = tl.constexpr(13)  # ceil(52 / _RADIX_BITS)


@triton.jit
def _encode_key(x, idx):
    """float32 score + int index -> deterministic sortable int64 key."""
    ui = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    neg = (ui >> 31) != 0
    # negative -> flip all bits, non-negative -> flip only the sign bit
    key32 = tl.where(neg, ui ^ 0xFFFFFFFF, ui ^ 0x80000000)
    return (key32 << _LOW_BITS) | (_LOW_MASK - idx.to(tl.int64))


@triton.jit
def _decode_key(key):
    """int64 key -> (float32 score, int32 index)."""
    idx = (_LOW_MASK - (key & _LOW_MASK)).to(tl.int32)
    key32 = key >> _LOW_BITS
    pos = (key32 >> 31) != 0  # top bit set in the remapped domain == original >= 0
    ui = tl.where(pos, key32 ^ 0x80000000, key32 ^ 0xFFFFFFFF)
    score = ui.to(tl.int32).to(tl.float32, bitcast=True)
    return score, idx


@triton.jit
def indexer_topk_kernel(
    scores_ptr,     # [q_len, k_len] float32
    counts_ptr,     # [q_len] int32, valid prefix length per row
    out_idx_ptr,    # [q_len, top_k] int32
    out_score_ptr,  # [q_len, top_k] float32
    k_len,
    top_k,
    stride_sq,
    stride_sk,
    K: tl.constexpr,  # next_pow2(top_k); also the streaming tile size
):
    row = tl.program_id(0)
    cnt = tl.load(counts_ptr + row).to(tl.int32)

    # Running top-K buffer, kept sorted descending. Init to -inf sentinels.
    buf = tl.full((K,), _SENTINEL, tl.int64)

    # Stream the row in tiles of K. For each tile we merge it into `buf`:
    # top-K of two descending length-K arrays A, B is
    #   sort( max(A[i], reverse(B)[i]) )       (bitonic top-K merge step)
    for start in range(0, k_len, K):
        offs = start + tl.arange(0, K)
        in_range = offs < k_len
        valid = offs < cnt
        x = tl.load(
            scores_ptr + row * stride_sq + offs * stride_sk,
            mask=in_range,
            other=float("-inf"),
        )
        x = tl.where(valid, x, float("-inf"))  # invalid tail -> never selected

        keys = tl.sort(_encode_key(x, offs), descending=True)
        cand = tl.maximum(buf, tl.flip(keys, 0))
        buf = tl.sort(cand, descending=True)

    score, idx = _decode_key(buf)
    o = tl.arange(0, K)
    # positions beyond the valid count get index -1 (scores stay -inf there)
    idx = tl.where(o < cnt, idx, -1)
    store_mask = o < top_k
    tl.store(out_idx_ptr + row * top_k + o, idx, mask=store_mask)
    tl.store(out_score_ptr + row * top_k + o, score, mask=store_mask)


def indexer_topk(scores: Tensor, counts: Tensor, top_k: int):
    """Deterministic, sorted top-k over the valid prefix of each row.

    scores: [q_len, k_len] float32 (row-major contiguous)
    counts: [q_len] int32, only scores[r, :counts[r]] are valid
    returns (topk_indices [q_len, top_k] int32, topk_scores [q_len, top_k] float32)
    """
    q_len, k_len = scores.shape
    K = triton.next_power_of_2(top_k)

    out_idx = paddle.empty([q_len, top_k], dtype="int32")
    out_score = paddle.empty([q_len, top_k], dtype="float32")

    grid = (q_len,)
    indexer_topk_kernel[grid](
        scores,
        counts,
        out_idx,
        out_score,
        k_len,
        top_k,
        k_len,  # stride_sq (contiguous)
        1,      # stride_sk
        K=K,
    )
    return out_idx, out_score


@triton.jit
def _radix_select_gather_kernel(
    scores_ptr,     # [q_len, k_len] float32
    counts_ptr,     # [q_len] int32
    scratch_ptr,    # [q_len, K] int64, holds the gathered top-k keys
    k_len,
    top_k,
    stride_sq,
    stride_sk,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Phase 1: MSD radix-select the top_k-th largest composite key (the exact
    split value), then gather the top_k qualifying keys into `scratch`.

    Because the composite key is a total order, the split value is unique and
    exactly top_k elements satisfy `key >= T`, so a running prefix-sum gives a
    dense placement without any tie handling."""
    row = tl.program_id(0)
    cnt = tl.load(counts_ptr + row).to(tl.int32)
    bins = tl.arange(0, _N_BINS)

    # ---- radix select: resolve the split key T bit-window by bit-window ----
    prefix = tl.zeros((), tl.int64)   # high bits fixed so far
    k_rem = top_k                     # rank still to locate within `prefix`
    for p in tl.static_range(_N_PASSES):
        shift = (_N_PASSES - 1 - p) * _RADIX_BITS
        hp = shift + _RADIX_BITS
        hist = tl.zeros((_N_BINS,), tl.int32)
        for start in range(0, k_len, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            in_range = offs < k_len
            x = tl.load(
                scores_ptr + row * stride_sq + offs * stride_sk,
                mask=in_range,
                other=float("-inf"),
            )
            x = tl.where(offs < cnt, x, float("-inf"))
            e = _encode_key(x, offs)
            match = ((e >> hp) == (prefix >> hp)) & in_range
            dig = ((e >> shift) & (_N_BINS - 1)).to(tl.int32)
            # tl.histogram does the block-wide bin count in one optimized op;
            # `match` drops elements that don't share the fixed prefix.
            hist += tl.histogram(dig, _N_BINS, mask=match)
        # walk buckets high -> low, pick the one holding the k_rem-th element
        ge = tl.flip(tl.cumsum(tl.flip(hist, 0), 0), 0)  # inclusive suffix sum
        gt = ge - hist                                   # strictly-higher buckets
        sel = (gt < k_rem) & (ge >= k_rem)               # exactly one bucket
        chosen = tl.sum(tl.where(sel, bins, 0))
        above = tl.sum(tl.where(sel, gt, 0))
        prefix = prefix | (chosen.to(tl.int64) << shift)
        k_rem = k_rem - above

    # ---- gather: compact all keys >= T into scratch[row, 0:top_k] ----
    split = prefix
    count = tl.zeros((), tl.int32)
    for start in range(0, k_len, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        in_range = offs < k_len
        x = tl.load(
            scores_ptr + row * stride_sq + offs * stride_sk,
            mask=in_range,
            other=float("-inf"),
        )
        x = tl.where(offs < cnt, x, float("-inf"))
        e = _encode_key(x, offs)
        qual = (e >= split) & in_range
        qi = qual.to(tl.int32)
        pos = count + tl.cumsum(qi, 0) - qi   # exclusive prefix + running count
        tl.store(scratch_ptr + row * K + pos, e, mask=qual & (pos < K))
        count += tl.sum(qi)

    # pad the unused tail (top_k .. K) with -inf sentinels for the sort phase
    tail = tl.arange(0, K)
    tl.store(
        scratch_ptr + row * K + tail,
        tl.full((K,), _SENTINEL, tl.int64),
        mask=tail >= count,
    )


@triton.jit
def _radix_sort_output_kernel(
    scratch_ptr,    # [q_len, K] int64
    counts_ptr,     # [q_len] int32
    out_idx_ptr,    # [q_len, top_k] int32
    out_score_ptr,  # [q_len, top_k] float32
    top_k,
    K: tl.constexpr,
):
    """Phase 2: sort the (at most K) gathered keys once and emit results."""
    row = tl.program_id(0)
    cnt = tl.load(counts_ptr + row).to(tl.int32)
    o = tl.arange(0, K)
    keys = tl.load(scratch_ptr + row * K + o)
    keys = tl.sort(keys, descending=True)
    score, idx = _decode_key(keys)
    idx = tl.where(o < cnt, idx, -1)
    m = o < top_k
    tl.store(out_idx_ptr + row * top_k + o, idx, mask=m)
    tl.store(out_score_ptr + row * top_k + o, score, mask=m)


def indexer_topk_radix(scores: Tensor, counts: Tensor, top_k: int):
    """Radix-select variant of `indexer_topk` with identical semantics.

    Finds the top_k split value with a multi-pass radix select, gathers the
    top_k elements, then sorts them once. Re-reads `scores` several times but
    avoids the repeated bitonic merges of the streaming kernel -- a better fit
    when `scores` is already materialized."""
    q_len, k_len = scores.shape
    K = triton.next_power_of_2(top_k)

    scratch = paddle.empty([q_len, K], dtype="int64")
    out_idx = paddle.empty([q_len, top_k], dtype="int32")
    out_score = paddle.empty([q_len, top_k], dtype="float32")

    grid = (q_len,)
    _radix_select_gather_kernel[grid](
        scores, counts, scratch,
        k_len, top_k,
        k_len, 1,
        K=K, BLOCK=1024,
    )
    _radix_sort_output_kernel[grid](
        scratch, counts, out_idx, out_score, top_k, K=K,
    )
    return out_idx, out_score


@triton.jit
def _merge_bufb(buf_a, base_ptr, K: tl.constexpr):
    """Merge the `nb` (<= K) pending keys in scratch[base_ptr, 0:nb] into the
    running top-K `buf_a` (both sorted descending). Returns the new buf_a.

    Exactly two sorts: sort buf_b, then the bitonic "max(A[i], reverse(B)[i])"
    merge with buf_a. Slots [nb, K) are padded with -inf sentinels first (a
    no-op when nb == K, i.e. buf_b is completely full)."""
    tl.debug_barrier()  # pending appends (this + prior tiles) now visible
    tail = tl.arange(0, K)
    b = tl.sort(tl.load(base_ptr + tail), descending=False)  # sort buf_b
    buf_a = tl.sort(tl.maximum(buf_a, b), descending=True)  # merge and sort
    tl.debug_barrier()  # finish reads before the caller overwrites buf_b
    return buf_a


@triton.jit
def _filtered_topk_kernel(
    scores_ptr,     # [q_len, k_len] float32
    counts_ptr,     # [q_len] int32
    scratch_ptr,    # [q_len, K] int64, pending buffer (buf_b)
    out_idx_ptr,    # [q_len, top_k] int32
    out_score_ptr,  # [q_len, top_k] float32
    k_len,
    top_k,
    stride_sq,
    stride_sk,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Filtered streaming top-K. Keep a sorted top-K in registers (buf_a) with
    threshold tau = min(buf_a); elements <= tau are dropped for free. Survivors
    are appended to a length-K pending buffer (buf_b); when a tile fills it past
    K we merge exactly the first K (2 sorts) and carry the overflow into the
    fresh buf_b for the next merge. As tau rises (~K/n acceptance), merges become
    rare and most tiles are pure read-and-discard."""
    row = tl.program_id(0)
    cnt = tl.load(counts_ptr + row).to(tl.int32)
    base = scratch_ptr + row * K

    buf_a = tl.full((K,), _SENTINEL, tl.int64)  # sorted-desc running top-K
    tau = tl.full((), _SENTINEL, tl.int64)      # == min(buf_a); accept key > tau
    nb = tl.zeros((), tl.int32)                 # pending count in buf_b

    for start in range(0, k_len, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        in_range = offs < k_len
        x = tl.load(
            scores_ptr + row * stride_sq + offs * stride_sk,
            mask=in_range,
            other=float("-inf"),
        )
        x = tl.where(offs < cnt, x, float("-inf"))
        e = _encode_key(x, offs)

        qual = (e > tau) & in_range
        qi = qual.to(tl.int32)
        dest = nb + tl.cumsum(qi, 0) - qi   # intended pending slot (pre-merge)
        total = nb + tl.sum(qi)

        if total >= K:
            # elements with dest < K fill buf_b up to K, then merge (2 sorts)
            tl.store(base + dest, e, mask=qual & (dest < K))
            buf_a = _merge_bufb(buf_a, base, K)
            tau = tl.min(buf_a, 0)
            # the overflow (dest >= K) carries into the fresh buf_b
            tl.store(base + (dest - K), e, mask=qual & (dest >= K))
            nb = total - K
        else:
            tl.store(base + dest, e, mask=qual)
            nb = total

    if nb > 0:  # flush the remaining tail (2 sorts)
        tail = tl.arange(0, K)
        tl.store(base + tail, tl.full((K,), _SENTINEL, tl.int64), mask=tail >= nb)
        buf_a = _merge_bufb(buf_a, base, K)

    score, idx = _decode_key(buf_a)
    o = tl.arange(0, K)
    idx = tl.where(o < cnt, idx, -1)
    m = o < top_k
    tl.store(out_idx_ptr + row * top_k + o, idx, mask=m)
    tl.store(out_score_ptr + row * top_k + o, score, mask=m)


def indexer_topk_filtered(scores: Tensor, counts: Tensor, top_k: int):
    """Filtered streaming variant of `indexer_topk` with identical semantics.

    Single read of `scores` with a rising threshold that discards most elements
    before they ever reach a sort. Best when scores are dynamically produced
    (read once) and roughly i.i.d. so the threshold climbs fast -> merges are
    logarithmic in k_len/top_k, each costing just two sorts."""
    q_len, k_len = scores.shape
    K = triton.next_power_of_2(top_k)

    scratch = paddle.empty([q_len, K], dtype="int64")
    out_idx = paddle.empty([q_len, top_k], dtype="int32")
    out_score = paddle.empty([q_len, top_k], dtype="float32")

    grid = (q_len,)
    _filtered_topk_kernel[grid](
        scores, counts, scratch, out_idx, out_score,
        k_len, top_k,
        k_len, 1,
        K=K, BLOCK=K,
    )
    return out_idx, out_score


def _numpy_ref(scores_np, counts_np, top_k):
    """Reference matching the required semantics: deterministic (smaller index
    wins ties), sorted descending, invalid tail masked to -inf / index -1.
    Uses a stable argsort instead of paddle.topk (which neither guarantees the
    tie-break nor reliably supports very large top_k)."""
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
    q_len, k_len, top_k = 64, 2048, 512

    scores = paddle.randn([q_len, k_len], "float32")
    counts = paddle.randint(0, k_len + 1, [q_len]).astype("int32")
    # exercise the boundary cases explicitly
    counts[0] = 0            # empty row -> all -1 / -inf
    counts[1] = 1            # fewer valid than top_k
    counts[2] = top_k        # exactly top_k valid
    counts[3] = k_len        # full row

    # idx, sc = indexer_topk(scores, counts, top_k)
    # idx, sc = indexer_topk_radix(scores, counts, top_k)
    idx, sc = indexer_topk_filtered(scores, counts, top_k)
    idx_np, sc_np = idx.numpy(), sc.numpy()

    ref_idx, ref_sc = _numpy_ref(scores.numpy(), counts.numpy(), top_k)

    idx_ok = np.array_equal(idx_np, ref_idx)
    sc_ok = np.array_equal(sc_np, ref_sc)  # bit-exact: we preserve fp32 bits
    print(f"[correctness] indices match ref: {idx_ok}")
    print(f"[correctness] scores  match ref: {sc_ok}")

    # determinism: identical bytes across independent runs
    idx2, sc2 = indexer_topk_filtered(scores, counts, top_k)
    det_ok = np.array_equal(idx2.numpy(), idx_np) and np.array_equal(
        sc2.numpy(), sc_np
    )
    print(f"[determinism] two runs identical: {det_ok}")

    # tie-break: many equal scores must resolve to the smaller original index
    tie_scores = paddle.to_tensor(
        np.tile(np.arange(8, dtype=np.float32) % 3, (4, 1)), "float32"
    )  # values repeat -> forces ties
    tie_counts = paddle.to_tensor([8, 8, 8, 8], "int32")
    t_idx, _ = indexer_topk_filtered(tie_scores, tie_counts, 8)
    r_idx, _ = _numpy_ref(tie_scores.numpy(), tie_counts.numpy(), 8)
    tie_ok = np.array_equal(t_idx.numpy(), r_idx)
    print(f"[tie-break]  smaller-index-wins matches ref: {tie_ok}")

    assert idx_ok and sc_ok and det_ok and tie_ok, "topk kernel verification FAILED"
    print("ALL CHECKS PASSED")
