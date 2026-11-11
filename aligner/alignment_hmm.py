# Fade Aligner. Derived from Stylish-TTS (https://github.com/Stylish-TTS/Stylish-TTS, MIT, Copyright (c) 2023 Aaron (Yinghao) Li).
# HMM decoding with CTC label priors follows Huang et al., ICASSP 2024 (arXiv 2406.02560). See NOTICE / LICENSE.
"""
alignment_hmm.py
----------------
Blank-free (HMM-style) monotonic alignment on the aligner's frame posteriors.

Every frame belongs to exactly one token, tokens keep text order, each token
takes >= 1 frame (or >= min_dur frames via repeated states), and tokens marked
skippable (e.g. word-boundary SIL) may take 0 frames (or, with skip_len > 1,
either 0 or >= skip_len frames).

  viterbi_durations()  best path -> per-token durations (sum == frames)
  forward_sum_loss()   -log sum over all such paths, normalised per frame;
                       gradient via forward-backward (no autograd through the
                       frame loop)
"""

import torch
import torch.nn.functional as F

NEG = -1e30


def nonblank_log_probs(log_probs, blank=0):
    """(N, T, C) log-probs -> log-probs renormalised over the non-blank classes."""
    lp = log_probs.float()
    mask = torch.zeros(lp.shape[-1], dtype=torch.bool, device=lp.device)
    mask[blank] = True
    lp = lp.masked_fill(mask, NEG)
    return lp - torch.logsumexp(lp, dim=-1, keepdim=True)


def expand(targets, target_lengths, skippable=None, min_dur=1, single_state=None, skip_len=1):
    """
    Repeat every token min_dur times (tokens in `single_state`, e.g. SIL, keep
    one state); skippable tokens get skip_len states that are skipped as a
    block. Returns state targets, state lengths, skippable mask over states
    and, per state, the index of its token.
    """
    N = targets.shape[0]
    rows_t, rows_tok, rows_skip = [], [], []
    for i in range(N):
        L = int(target_lengths[i])
        st, tok, sk = [], [], []
        for j in range(L):
            t = int(targets[i, j])
            skip = skippable is not None and bool(skippable[i, j])
            if skip:
                reps = skip_len
            else:
                reps = 1 if (min_dur <= 1 or (single_state is not None and t in single_state)) else min_dur
            for _ in range(reps):
                st.append(t); tok.append(j); sk.append(skip)
        rows_t.append(st); rows_tok.append(tok); rows_skip.append(sk)
    S = max(1, max(len(r) for r in rows_t))
    st = torch.zeros(N, S, dtype=torch.long)
    tok = torch.full((N, S), -1, dtype=torch.long)
    sk = torch.zeros(N, S, dtype=torch.bool)
    lens = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        n = len(rows_t[i]); lens[i] = n
        if n:
            st[i, :n] = torch.tensor(rows_t[i]); tok[i, :n] = torch.tensor(rows_tok[i])
            sk[i, :n] = torch.tensor(rows_skip[i])
    return st, lens, sk, tok


def emissions(lp, state_targets, state_lengths):
    """(N, T, C), (N, S) -> (N, T, S); padded states are NEG."""
    N, T, _ = lp.shape
    S = state_targets.shape[1]
    e = torch.gather(lp, 2, state_targets.to(lp.device).unsqueeze(1).expand(N, T, S))
    pad = torch.arange(S, device=lp.device)[None, :] >= state_lengths.to(lp.device)[:, None]
    return e.masked_fill(pad[:, None, :], NEG)


def _skip_into(skippable, skip_len=1):
    """State s can be entered from the state before the skippable block that ends at s-1."""
    N, S = skippable.shape
    run = torch.zeros(N, S, dtype=torch.long, device=skippable.device)
    r = torch.zeros(N, dtype=torch.long, device=skippable.device)
    for s in range(S):
        r = torch.where(skippable[:, s], r + 1, torch.zeros_like(r))
        run[:, s] = r
    block_end = skippable & (run % skip_len == 0)
    return F.pad(block_end, (1, 0), value=False)[:, :S]


def _lse3(a, b, c):
    m = torch.maximum(torch.maximum(a, b), c)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m) + torch.exp(c - m))


def _cands(prev, skip_into, skip_len=1):
    S = prev.shape[1]
    adv = F.pad(prev, (1, 0), value=NEG)[:, :S]
    skp = F.pad(prev, (skip_len + 1, 0), value=NEG)[:, :S].masked_fill(~skip_into, NEG)
    return torch.stack([prev, adv, skp], -1)


def _init(emit, skippable, skip_len=1):
    N, _, S = emit.shape
    d = torch.full((N, S), NEG, dtype=emit.dtype, device=emit.device)
    d[:, 0] = emit[:, 0, 0]
    if S > skip_len:
        d[:, skip_len] = torch.where(skippable[:, 0], emit[:, 0, skip_len],
                                     torch.full_like(emit[:, 0, skip_len], NEG))
    return d


def _final_mask(state_lengths, skippable, S, skip_len=1):
    rows = torch.arange(len(state_lengths))
    last = (state_lengths - 1).clamp(min=0)
    m = torch.zeros(len(state_lengths), S, dtype=torch.bool)
    m[rows, last] = True
    prev = (state_lengths - 1 - skip_len).clamp(min=0)
    early = skippable[rows, last] & (state_lengths >= skip_len + 1)
    m[rows[early], prev[early]] = True
    return m


@torch.no_grad()
def viterbi_durations(emit, in_len, state_lengths, skippable, skip_len=1):
    """emit (N, T, S) -> durations (N, S) long, feasible (N,) bool."""
    emit = emit.float()
    N, T, S = emit.shape
    dev = emit.device
    in_len = in_len.to(dev).long(); skippable = skippable.to(dev)
    skip_into = _skip_into(skippable, skip_len)
    back = torch.zeros(N, T, S, dtype=torch.int8, device=dev)
    step = torch.tensor([0, 1, skip_len + 1], device=dev)
    d = _init(emit, skippable, skip_len)
    for t in range(1, T):
        best, arg = _cands(d, skip_into, skip_len).max(-1)
        d = torch.where((t < in_len)[:, None], best + emit[:, t], d)
        back[:, t] = arg.to(torch.int8)
    fm = _final_mask(state_lengths.cpu(), skippable.cpu(), S, skip_len).to(dev)
    end_scores = d.masked_fill(~fm, NEG)
    score, state = end_scores.max(-1)
    feasible = score > NEG / 2
    rows = torch.arange(N, device=dev)
    path = torch.full((N, T), -1, dtype=torch.long, device=dev)
    for t in range(T - 1, -1, -1):
        active = t < in_len
        path[:, t] = torch.where(active, state, path[:, t])
        if t > 0:
            state = torch.where(active, state - step[back[rows, t, state].long()], state)
    dur = torch.zeros(N, S, dtype=torch.long, device=dev)
    dur.scatter_add_(1, path.clamp(min=0), (path >= 0).long())
    dur[~feasible] = 0
    return dur, feasible


class _ForwardSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, emit, in_len, state_lengths, skippable, skip_len=1):
        emit_d = emit.detach().float()
        N, T, S = emit_d.shape
        dev = emit_d.device
        k = skip_len + 1
        skip_into = _skip_into(skippable, skip_len)
        alpha = torch.full((N, T, S), NEG, dtype=torch.float32, device=dev)
        a = _init(emit_d, skippable, skip_len)
        alpha[:, 0] = a
        for t in range(1, T):
            adv = F.pad(a, (1, 0), value=NEG)[:, :S]
            skp = F.pad(a, (k, 0), value=NEG)[:, :S].masked_fill(~skip_into, NEG)
            a_new = _lse3(a, adv, skp) + emit_d[:, t]
            a = torch.where((t < in_len)[:, None], a_new, a)
            alpha[:, t] = a
        fm = _final_mask(state_lengths.cpu(), skippable.cpu(), S, skip_len).to(dev)
        ll = torch.logsumexp(a.masked_fill(~fm, NEG), -1)
        # backward pass (beta)
        beta = torch.full((N, T, S), NEG, dtype=torch.float32, device=dev)
        b = torch.full((N, S), NEG, dtype=torch.float32, device=dev)
        skip_from = F.pad(skip_into, (0, k), value=False)[:, k:]     # s -> s+k allowed
        for t in range(T - 1, -1, -1):
            is_last = (t == in_len - 1)[:, None]
            nxt = emit_d[:, t + 1] + b if t + 1 < T else torch.full_like(b, NEG)
            stay = nxt
            adv = F.pad(nxt, (0, 1), value=NEG)[:, 1:]
            skp = F.pad(nxt, (0, k), value=NEG)[:, k:].masked_fill(~skip_from, NEG)
            rec = _lse3(stay, adv, skp)
            b = torch.where(is_last, torch.where(fm, torch.zeros_like(b), torch.full_like(b, NEG)),
                            torch.where((t < in_len - 1)[:, None], rec, b))
            beta[:, t] = b
        gamma = torch.exp(alpha + beta - ll[:, None, None])
        gamma = torch.where(torch.arange(T, device=dev)[None, :, None] < in_len[:, None, None], gamma,
                            torch.zeros_like(gamma))
        ctx.save_for_backward(gamma)
        return ll.to(emit.dtype)

    @staticmethod
    def backward(ctx, grad_ll):
        (gamma,) = ctx.saved_tensors
        return (grad_ll.float()[:, None, None] * gamma).to(grad_ll.dtype), None, None, None, None


def forward_sum_ll(emit, in_len, state_lengths, skippable, skip_len=1):
    dev = emit.device
    return _ForwardSum.apply(emit, in_len.to(dev).long(), state_lengths.to(dev).long(), skippable.to(dev), skip_len)
