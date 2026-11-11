# Fade Aligner. Derived from Stylish-TTS (https://github.com/Stylish-TTS/Stylish-TTS, MIT, Copyright (c) 2023 Aaron (Yinghao) Li).
# HMM decoding with CTC label priors follows Huang et al., ICASSP 2024 (arXiv 2406.02560). See NOTICE / LICENSE.
import os
import os.path as osp
import logging
import argparse
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import k2
import torchaudio
from typing import Tuple, Union, List
from torch.utils.data import DataLoader

from alignment_hmm import nonblank_log_probs, expand, emissions, forward_sum_ll, viterbi_durations

# Import dataset preparation
# Ensure meldataset_preparation.py is in the same directory or python path
from meldataset_preparation import (
    build_dataloader,
    symbols,
    BLANK_TOKEN_ID,
    MEL_PARAMS,
    SPOKEN_IDS,
    PAD_ID,
    SIL_ID,
    SPECT_PARAMS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# MODEL DEFINITIONS (Extracted from stylish-tts)
# -----------------------------------------------------------------------------

class CTCModel(torch.nn.Module):
    r"""
    This implements a CTC model with an encoder and a projection layer
    """

    def __init__(self, encoder, encoder_output_layer, n_token, n_mels) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_output_layer = encoder_output_layer

    def ctc_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            The output tensor from the transformer encoder.
            Its shape is (B, T', D')

        Returns:
          Return a tensor that can be used for CTC decoding.
          Its shape is (B, T, V), where V is the number of classes
        """
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        x = nn.functional.log_softmax(x, dim=-1)  # (T, N, C)
        return x

    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        device = self.encoder_output_layer.weight.device
        if sources.device != device:
            sources = sources.to(device)
            source_lengths = source_lengths.to(device)

        source_encodings, source_lengths = self.encoder(
            input=sources,
            lengths=source_lengths,
        )

        posterior = self.encoder_output_layer(source_encodings)
        ctc_log_prob = self.ctc_output(posterior)

        # Return source_lengths so callers use the true encoder output lengths
        # (not the raw mel lengths), which is the correct input_lengths for CTC.
        return ctc_log_prob, source_lengths.long()

class Ffn(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, out_dim, nlayers=1, drop_out=0.1, skip=False
    ) -> None:
        super().__init__()

        layers = []
        for ilayer in range(nlayers):
            _in = hidden_dim if ilayer > 0 else input_dim
            _out = hidden_dim if ilayer < nlayers - 1 else out_dim
            layers.extend(
                [
                    nn.Linear(_in, _out),
                    nn.ReLU(),
                    nn.Dropout(p=drop_out),
                ]
            )
        self.ffn = torch.nn.Sequential(
            *layers,
        )

        self.skip = skip

    def forward(self, x) -> torch.Tensor:
        x_out = self.ffn(x)

        if self.skip:
            x_out = x_out + x

        return x_out

class TdnnBlstm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim=640,
        drop_out=0.1,
        tdnn_blstm_spec=[],
    ) -> None:
        super().__init__()

        self.tdnn_blstm_spec = tdnn_blstm_spec

        layers = nn.ModuleList([])
        layers_info = []
        for i_layer, spec in enumerate(tdnn_blstm_spec):
            if spec[0] == "tdnn":
                ll = []
                dilation = spec[3] if len(spec) >= 4 else 1
                padding = int((spec[1] - 1) / 2) * dilation
                ll.append(
                    nn.Conv1d(
                        in_channels=input_dim if len(layers) == 0 else hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=spec[1],  # 3
                        dilation=dilation,
                        stride=spec[2],  # 1
                        padding=padding,  # 1
                    )
                )
                ll.append(nn.ReLU(inplace=True))
                ll.append(nn.BatchNorm1d(num_features=hidden_dim, affine=False))
                if drop_out > 0:
                    ll.append(nn.Dropout(drop_out))

                layers.append(nn.Sequential(*ll))
                layers_info.append(("tdnn", spec))

            elif spec[0] == "blstm":
                raise NotImplementedError("BLSTM layer not supported in this standalone script")

            elif spec[0] == "ffn":
                layers.append(
                    Ffn(
                        input_dim=input_dim if len(layers) == 0 else hidden_dim,
                        hidden_dim=hidden_dim,
                        out_dim=hidden_dim,
                        skip=True,
                        drop_out=drop_out,
                        nlayers=spec[1],
                    )
                )
                layers_info.append(("ffn", spec))

        self.layers = layers
        self.layers_info = layers_info

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = input
        for layer, (layer_type, spec) in zip(self.layers, self.layers_info):
            if layer_type == "tdnn":
                mask = (
                    torch.arange(lengths.max(), device=x.device)[None, :]
                    < lengths[:, None]
                ).float()
                x = x * mask.unsqueeze(2)  # masking/padding
                x = x.permute(0, 2, 1)  # (N, T, C) ->(N, C, T)
                x = layer(x)
                x = x.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

                stride = spec[2]
                if True:  # stride > 1:
                    kernel_size = spec[1]
                    dilation = spec[3] if len(spec) >= 4 else 1
                    padding = int((spec[1] - 1) / 2) * dilation
                    lengths = lengths + 2 * padding - dilation * (kernel_size - 1) - 1
                    lengths = lengths / stride + 1
                    lengths = torch.floor(lengths)
            elif layer_type == "blstm":
                 raise NotImplementedError("BLSTM layer not supported")
            else:
                x = layer(x)
        return x, lengths


def tdnn_blstm_ctc_model(
    input_dim: int, num_symbols: int, hidden_dim=640, drop_out=0.1, tdnn_blstm_spec=[]
):
    """Builds TDNN-BLSTM-based CTC model. Matches stylish-tts function signature."""
    return CTCModel(
        encoder=TdnnBlstm(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drop_out=drop_out,
            tdnn_blstm_spec=tdnn_blstm_spec,
        ),
        encoder_output_layer=nn.Linear(hidden_dim, num_symbols),  # blank already included at ID 0
        n_token=num_symbols,
        n_mels=input_dim,
    )


def tdnn_blstm_ctc_model_base(n_mels, num_symbols):
    """
    1:1 match of stylish-tts tdnn_blstm_ctc_model_base.
    Spec: exactly [("tdnn",5,1,1), ("tdnn",3,1,1), ("tdnn",3,1,1), ("ffn",5)]
    hidden_dim=640, drop_out=0.1
    """
    return tdnn_blstm_ctc_model(
        input_dim=n_mels,
        num_symbols=num_symbols,
        hidden_dim=640,
        drop_out=0.1,
        tdnn_blstm_spec=[
            ("tdnn", 5, 1, 1),
            ("tdnn", 3, 1, 1),
            ("tdnn", 3, 1, 1),
            ("ffn", 5),
        ],
    )

# -----------------------------------------------------------------------------
# LOSS FUNCTION — 1:1 from stylish-tts/src/stylish_tts/train/losses.py
# Reference: https://github.com/huangruizhe/audio/blob/aligner_label_priors/examples/asr/librispeech_alignment/loss.py
# -----------------------------------------------------------------------------

class CTCLossWithLabelPriors(nn.Module):
    def __init__(self, prior_scaling_factor=0.0, blank=0, reduction="mean"):
        super().__init__()

        self.blank = blank
        self.reduction = reduction

        self.log_priors = None
        self.log_priors_sum = None
        self.num_samples = 0
        self.prior_scaling_factor = prior_scaling_factor
        self.k2_device = "cpu"

    def to(self, device):
        super().to(device)
        target_device = torch.device(device)
        use_k2_cuda = target_device.type == "cuda" and bool(getattr(k2, "with_cuda", False))
        self.k2_device = str(target_device) if use_k2_cuda else "cpu"
        if target_device.type == "cuda" and not use_k2_cuda:
            logger.warning("k2 was built without CUDA support; CTC loss will run on CPU.")
        return self

    def encode_supervisions(
        self, targets, target_lengths, input_lengths
    ) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]], torch.Tensor]:
        # https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py#L181

        batch_size = targets.size(0)
        supervision_segments = torch.stack(
            (
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]

        # Be careful: the targets here are already padded! We need to remove paddings from it.
        res = targets[indices].tolist()
        res_lengths = target_lengths[indices].tolist()
        res = [l[:l_len] for l, l_len in zip(res, res_lengths)]

        return supervision_segments, res, indices

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        step_type="train",
    ) -> Tensor:
        supervision_segments, token_ids, indices = self.encode_supervisions(
            targets, target_lengths, input_lengths
        )
        target_lengths_sorted = target_lengths[indices].to(self.k2_device)

        decoding_graph = k2.ctc_graph(token_ids, modified=False, device=self.k2_device)

        # Permute: (T, N, C) -> (N, T, C) — log_probs stay in original batch order;
        # supervision_segments carries the original batch indices so k2 picks the right rows.
        log_probs = log_probs.permute(1, 0, 2)

        # Accumulate label priors for this epoch
        if step_type == "train":
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[: int(le.item())])
            log_probs_flattened = torch.cat(log_probs_flattened, 0)

            # Note: log_probs is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.num_samples += T
            log_batch_priors_sum = torch.logsumexp(
                log_probs_flattened, dim=0, keepdim=True
            )
            log_batch_priors_sum = log_batch_priors_sum.detach()
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

            # Apply the label priors
            if self.log_priors is not None and self.prior_scaling_factor > 0:
                log_probs = log_probs - self.log_priors * self.prior_scaling_factor

        # Compute CTC loss
        dense_fsa_vec = k2.DenseFsaVec(
            log_probs.to(self.k2_device),          # (N, T, C)
            supervision_segments.cpu(),
        )

        loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=10,
            reduction=self.reduction,
            use_double_scores=True,
            target_lengths=target_lengths_sorted,
        )

        return loss.to(log_probs.device)

    def forced_align(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ):
        """
        k2-based batched forced alignment.
        log_probs: (N, T, C) — already permuted before calling.
        Returns: (batch_frame_labels, scores)
        """
        supervision_segments, token_ids, indices = self.encode_supervisions(
            targets, target_lengths, input_lengths
        )
        target_lengths_sorted = target_lengths[indices].to(self.k2_device)

        decoding_graph = k2.ctc_graph(token_ids, modified=False, device=self.k2_device)

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs.to(self.k2_device),
            supervision_segments.cpu(),
        )
        lattices = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            output_beam=10,
        )

        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        tot_scores = best_paths.get_tot_scores(
            log_semiring=True, use_double_scores=True
        )
        scores = tot_scores / target_lengths_sorted.to(tot_scores.device)

        batch_arc_shape = best_paths.arcs.shape().remove_axis(1)
        batch_frame_labels = k2.RaggedTensor(
            batch_arc_shape, best_paths.aux_labels
        ).tolist()
        # k2 makes an extra frame for some reasons
        for i in range(len(batch_frame_labels)):
            batch_frame_labels[i][-1] -= 1

        # Unsort alignment outputs back to original batch order.
        batch_size = targets.size(0)
        inverse_indices = torch.empty_like(indices)
        inverse_indices[indices] = torch.arange(batch_size, device=indices.device)
        inverse_indices_list = inverse_indices.cpu().tolist()
        batch_frame_labels = [
            batch_frame_labels[inverse_indices_list[i]] for i in range(batch_size)
        ]
        scores = scores[inverse_indices.to(scores.device)]
        return batch_frame_labels, scores

    def on_train_epoch_end(self):
        """
        Update label priors at end of each epoch.
        Single-GPU simplified version of stylish-tts on_train_epoch_end
        (no accelerator.gather needed).
        """
        if self.log_priors_sum is not None:
            log_priors_sums = self.log_priors_sum
            num_samples_log = torch.log(
                torch.tensor([self.num_samples], dtype=log_priors_sums.dtype,
                              device=log_priors_sums.device) + 1e-9
            )
            new_log_prior = log_priors_sums - num_samples_log

            prior_threshold = -12.0
            new_log_prior = torch.where(
                new_log_prior < prior_threshold,
                torch.tensor(prior_threshold, dtype=new_log_prior.dtype,
                              device=new_log_prior.device),
                new_log_prior,
            )

            self.log_priors = new_log_prior
            self.log_priors_sum = None
            self.num_samples = 0


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

class LossLog:
    def __init__(self):
        self._losses = {}
    def add_loss(self, name: str, value: torch.Tensor):
        self._losses[name] = value
    def get(self, name: str, default: float = 0.0) -> float:
        v = self._losses.get(name)
        if v is None:
            return default
        return v.item() if torch.is_tensor(v) else float(v)
    def __str__(self):
        return ", ".join(f"{k}: {float(v):.4f}" for k, v in self._losses.items())

def filter_batch_targets(text: torch.Tensor, text_lengths: torch.Tensor):
    """
    Filters text to keep only spoken characters.
    Padding of the returned dense tensor is controlled by lengths only.
    """
    device = text.device
    B = text.size(0)
    allowed_ids = set(SPOKEN_IDS)
    filtered_list = []
    for i in range(B):
        curr_len = int(text_lengths[i].item())
        t = text[i, :curr_len]
        t_filtered = [token.item() for token in t if token.item() in allowed_ids]
        filtered_list.append(torch.tensor(t_filtered, dtype=torch.long, device=device))
        
    new_lengths = torch.tensor([len(t) for t in filtered_list], dtype=torch.long, device=device)
    max_len = new_lengths.max().item() if new_lengths.numel() > 0 else 0
    if max_len == 0: max_len = 1 
    new_text = torch.full((B, max_len), fill_value=BLANK_TOKEN_ID, dtype=torch.long, device=device)
    for i, t in enumerate(filtered_list):
        if len(t) > 0: new_text[i, :len(t)] = t
    return new_text, new_lengths

# -----------------------------------------------------------------------------
# TRAINING LOGIC — 1:1 with stylish-tts train_alignment / validate_alignment
# from stylish-tts/src/stylish_tts/train/stage_type.py
# -----------------------------------------------------------------------------

def inner_sil_mask(text_filtered, text_length_filtered):
    """Word-boundary SILs (not the leading / trailing one) may take 0 frames."""
    pos = torch.arange(text_filtered.shape[1], device=text_filtered.device)[None, :]
    return (text_filtered == SIL_ID) & (pos > 0) & (pos < text_length_filtered[:, None] - 1)


def forward_sum_loss(ctc, text_filtered, enc_lengths, text_length_filtered):
    """-log P(text | audio) over blank-free monotonic paths (every frame belongs
    to one token, inner SILs optional), per frame, on the non-blank posteriors."""
    lp = nonblank_log_probs(ctc.permute(1, 0, 2))
    st, st_len, skip, _ = expand(text_filtered.cpu(), text_length_filtered.cpu(),
                                 inner_sil_mask(text_filtered, text_length_filtered).cpu())
    ll = forward_sum_ll(emissions(lp, st.to(lp.device), st_len), enc_lengths.long(), st_len, skip)
    ok = ll > -1e29
    if not ok.any():
        return ll.sum() * 0.0
    return (-ll[ok] / enc_lengths[ok].float()).mean()


# -----------------------------------------------------------------------------
# VITERBI TRAINING — frame-level cross-entropy on the aligner's own alignments,
# then realign with the blank-free HMM Viterbi (classic HMM-DNN realignment).
# -----------------------------------------------------------------------------

def ctc_midpoint_durations(ctc_loss_fn, log_probs, targets, enc_lengths, target_lengths):
    """Token durations from CTC forced alignment, boundaries halfway between
    consecutive token spikes (the base k2_align rule, without SIL heuristics)."""
    labels, _ = ctc_loss_fn.forced_align(log_probs=log_probs, targets=targets,
                                         input_lengths=enc_lengths, target_lengths=target_lengths)
    out = []
    for i, seq in enumerate(labels):
        T, L = int(enc_lengths[i]), int(target_lengths[i])
        onsets = [t for t, lab in enumerate(seq[:T]) if lab > 0]
        if L == 0 or len(onsets) != L:
            out.append(None)
            continue
        bounds = [0] + [(onsets[j] + onsets[j + 1] + 1) // 2 for j in range(L - 1)] + [T]
        out.append(torch.tensor([bounds[j + 1] - bounds[j] for j in range(L)]))
    return out


_NEG = -1e30


def ctc_viterbi_optional(lp, tokens, optional):
    """
    lp: (T, C) log-probs, tokens: list of ids, optional: list of bools (token may be skipped;
    two adjacent optional tokens cannot both be skipped — never the case for word-boundary SILs).
    Returns (onset frame or None per token, best path score).
    """
    T = lp.shape[0]
    L = len(tokens)
    S = 2 * L + 1
    ext = [0] * S
    for j, t in enumerate(tokens):
        ext[2 * j + 1] = t
    # allowed predecessor offsets per state (0..4)
    allow = torch.zeros(S, 5, dtype=torch.bool)
    allow[:, 0] = True
    allow[1:, 1] = True
    for s in range(2, S):
        if ext[s] != 0 and ext[s] != ext[s - 2]:
            allow[s, 2] = True
    for j, opt in enumerate(optional):
        if not opt:
            continue
        o = 2 * j + 1
        if o + 1 < S:
            allow[o + 1, 2] = True                           # blank before SIL -> blank after
            if o - 2 >= 0:
                allow[o + 1, 3] = True                       # token before SIL -> blank after
        if o + 2 < S:
            allow[o + 2, 3] = True                           # blank before SIL -> next token
            if o - 2 >= 0 and ext[o - 2] != ext[o + 2]:
                allow[o + 2, 4] = True                       # token before SIL -> next token
    emit = lp[:, torch.tensor(ext)].double()                 # (T, S)
    delta = torch.full((S,), _NEG, dtype=torch.float64)
    delta[0] = emit[0, 0]
    if S > 1:
        delta[1] = emit[0, 1]
    back = torch.zeros(T, S, dtype=torch.int8)
    pads = [torch.full((k,), _NEG, dtype=torch.float64) for k in range(5)]
    for t in range(1, T):
        cands = torch.stack([torch.cat([pads[k], delta[:S - k]]) if k else delta for k in range(5)], 1)
        cands = cands.masked_fill(~allow, _NEG)
        best, arg = cands.max(1)
        delta = best + emit[t]
        back[t] = arg.to(torch.int8)
    finals = [S - 1, S - 2]
    if L >= 1 and optional[-1]:
        finals += [S - 3, S - 4]
    finals = [f for f in finals if f >= 0]
    s = max(finals, key=lambda f: float(delta[f]))
    score = float(delta[s])
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    onsets = [None] * L
    for t in range(T):
        st = path[t]
        if st % 2 == 1 and onsets[st // 2] is None:
            onsets[st // 2] = t
    return onsets, score



def _optional_allow(tokens, optional):
    """allowed predecessor offsets (S, 5) of ctc_viterbi_optional's extended state sequence"""
    L = len(tokens)
    S = 2 * L + 1
    ext = [0] * S
    for j, t in enumerate(tokens):
        ext[2 * j + 1] = t
    allow = torch.zeros(S, 5, dtype=torch.bool)
    allow[:, 0] = True
    allow[1:, 1] = True
    for s in range(2, S):
        if ext[s] != 0 and ext[s] != ext[s - 2]:
            allow[s, 2] = True
    for j, opt in enumerate(optional):
        if not opt:
            continue
        o = 2 * j + 1
        if o + 1 < S:
            allow[o + 1, 2] = True
            if o - 2 >= 0:
                allow[o + 1, 3] = True
        if o + 2 < S:
            allow[o + 2, 3] = True
            if o - 2 >= 0 and ext[o - 2] != ext[o + 2]:
                allow[o + 2, 4] = True
    return ext, allow


@torch.no_grad()
def ctc_viterbi_optional_batch(log_probs, jobs):
    """
    ctc_viterbi_optional for a whole batch at once: log_probs (B, T, C) on any device (GPU: one step per frame for
    all utterances instead of a Python loop per utterance), jobs {i: (T_i, tokens, optional)}.
    Same recursion in float64, so the same paths. Returns {i: onsets}.
    """
    if not jobs:
        return {}
    dev = log_probs.device
    idx = sorted(jobs)
    Ts = [jobs[i][0] for i in idx]
    Ss = [2 * len(jobs[i][1]) + 1 for i in idx]
    B, Tm, Sm = len(idx), max(Ts), max(Ss)
    ext = torch.zeros(B, Sm, dtype=torch.long)
    allow = torch.zeros(B, Sm, 5, dtype=torch.bool)
    for b, i in enumerate(idx):
        e, a = _optional_allow(jobs[i][1], jobs[i][2])
        ext[b, :Ss[b]] = torch.tensor(e)
        allow[b, :Ss[b]] = a
    ext, allow = ext.to(dev), allow.to(dev)
    lp = log_probs[torch.tensor(idx, device=dev), :Tm].double()          # (B, Tm, C)
    emit = lp.gather(2, ext[:, None, :].expand(B, Tm, Sm))               # (B, Tm, Sm)
    T_t = torch.tensor(Ts, device=dev)
    delta = torch.full((B, Sm), _NEG, dtype=torch.float64, device=dev)
    delta[:, 0] = emit[:, 0, 0]
    delta[:, 1] = emit[:, 0, 1]
    back = torch.zeros(Tm, B, Sm, dtype=torch.int8, device=dev)
    pad = torch.full((B, 4), _NEG, dtype=torch.float64, device=dev)
    for t in range(1, Tm):
        dp = torch.cat([pad, delta], 1)
        cands = torch.stack([dp[:, 4 - k:4 - k + Sm] for k in range(5)], 2).masked_fill(~allow, _NEG)
        best, arg = cands.max(2)
        live = (t < T_t)[:, None]
        delta = torch.where(live, best + emit[:, t], delta)
        back[t] = torch.where(live, arg, 0).to(torch.int8)
    delta, back = delta.cpu(), back.cpu()
    s = torch.zeros(B, dtype=torch.long)
    for b, i in enumerate(idx):
        S, optional = Ss[b], jobs[i][2]
        finals = [S - 1, S - 2]
        if optional and optional[-1]:
            finals += [S - 3, S - 4]
        s[b] = max([f for f in finals if f >= 0], key=lambda f: float(delta[b, f]))
    path = torch.zeros(B, Tm, dtype=torch.long)
    ar = torch.arange(B)
    for t in range(Tm - 1, -1, -1):
        path[:, t] = s
        s = s - back[t, ar, s].long()
    out = {}
    for b, i in enumerate(idx):
        L = len(jobs[i][1])
        p = path[b, :Ts[b]]
        onsets = [None] * L
        odd = (p % 2 == 1).nonzero().squeeze(1)
        if len(odd):
            st = p[odd]
            first = torch.full((Ss[b],), -1, dtype=torch.long).scatter_reduce(0, st, odd, "amin", include_self=False)
            for j in range(L):
                if first[2 * j + 1] >= 0:
                    onsets[j] = int(first[2 * j + 1])
        out[i] = onsets
    return out


PUNCTUATION_TOKENS = set(';:,.!?¡¿—–…"«»"“”')


def ctc_optsil_durations(log_probs, tf, tl, text_full, text_length_full, enc_lengths):
    """
    Token durations from CTC alignment where word-boundary SILs without
    punctuation may take 0 frames (no fake micro-pauses), SILs after
    punctuation stay mandatory, and boundaries next to a SIL go where the
    posteriors hand over (instead of the spike midpoint).
    """
    jobs = {}
    for i in range(tf.shape[0]):
        L, T = int(tl[i]), int(enc_lengths[i])
        if L == 0 or T < L:
            continue
        tokens = tf[i, :L].tolist()
        full_tokens = text_full[i, :int(text_length_full[i])].tolist()
        spoken = [p for p, t in enumerate(full_tokens) if t in SPOKEN_IDS]
        after_punct = [j > 0 and any(symbols[full_tokens[q]] in PUNCTUATION_TOKENS
                                     for q in range(spoken[j - 1] + 1, spoken[j])) for j in range(L)]
        jobs[i] = (T, tokens, [tokens[j] == SIL_ID and 0 < j < L - 1 and not after_punct[j] for j in range(L)])
    # the whole batch in one Viterbi pass on the log-probs' device (same DP as ctc_viterbi_optional)
    onsets_all = ctc_viterbi_optional_batch(log_probs, jobs)
    log_probs = log_probs.float().cpu()
    out = []
    for i in range(tf.shape[0]):
        if i not in jobs:
            out.append(None)
            continue
        T, tokens, _ = jobs[i]
        L = len(tokens)
        onsets = onsets_all[i]
        emitted = [j for j in range(L) if onsets[j] is not None]
        if not emitted:
            out.append(None)
            continue
        bounds = [0]
        for a, b in zip(emitted, emitted[1:]):
            lo, hi = onsets[a] + 1, onsets[b]
            if SIL_ID in (tokens[a], tokens[b]) and hi > lo:
                cum_a = torch.cat([torch.zeros(1), log_probs[i, lo:hi, tokens[a]].cumsum(0)])
                cum_b = torch.cat([torch.zeros(1), log_probs[i, lo:hi, tokens[b]].cumsum(0)])
                bounds.append(lo + int((cum_a + (cum_b[-1] - cum_b)).argmax()))
            else:
                bounds.append((onsets[a] + onsets[b] + 1) // 2)
        bounds.append(T)
        d = torch.zeros(L, dtype=torch.long)
        for k, j in enumerate(emitted):
            d[j] = bounds[k + 1] - bounds[k]
        out.append(d)
    return out


@torch.no_grad()
def frame_labels(model, loader, ctc_loss_fn, device, use_hmm, min_dur=1, optional_sil=True):
    """name -> (T,) frame labels (token ids) from the current model."""
    model.eval()
    labels = {}
    for batch in loader:
        if batch is None:
            continue
        mel_length = batch.mel_length.to(device)
        tf, tl = filter_batch_targets(batch.text.to(device), batch.text_length.to(device))
        ctc, enc = model(batch.mels.to(device).permute(0, 2, 1), mel_length)
        log_probs = ctc.permute(1, 0, 2).contiguous()
        if use_hmm:
            lp = nonblank_log_probs(log_probs)
            skippable = inner_sil_mask(tf, tl) if optional_sil else torch.zeros_like(tf, dtype=torch.bool)
            st, st_len, skip, tok = expand(tf.cpu(), tl.cpu(), skippable.cpu(),
                                           min_dur=min_dur, single_state={SIL_ID})
            st_dur, ok = viterbi_durations(emissions(lp, st.to(lp.device), st_len), enc.long(), st_len, skip)
            durs = []
            for i in range(len(batch.names)):
                if not bool(ok[i]):
                    durs.append(None)
                    continue
                L = int(tl[i]); n = int(st_len[i])
                d = torch.zeros(L, dtype=torch.long)
                d.scatter_add_(0, tok[i, :n], st_dur[i, :n].cpu())
                durs.append(d)
        else:
            durs = ctc_optsil_durations(log_probs, tf, tl, batch.text, batch.text_length, enc.long())
        for i, name in enumerate(batch.names):
            if durs[i] is not None:
                L = int(tl[i])
                labels[name] = torch.repeat_interleave(tf[i, :L].cpu(), durs[i]).to(device)  # kept on the GPU for the CE epochs
    model.train()
    return labels


def train_ce_epoch(model, loader, optimizer, labels, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        ctc, enc = model(batch.mels.to(device).permute(0, 2, 1), batch.mel_length.to(device))
        log_probs = ctc.permute(1, 0, 2)                 # (N, T, C)
        lp, lab = [], []
        for i, name in enumerate(batch.names):
            if name in labels:
                y = labels[name].to(device)
                T = min(int(enc[i]), len(y))
                lp.append(log_probs[i, :T]); lab.append(y[:T])
        if not lp:
            continue
        loss = F.nll_loss(torch.cat(lp), torch.cat(lab))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def train_one_epoch(model, dataloader, optimizer, scaler, ctc_loss_fn, device, epoch, num_epochs,
                    ctc_weight=1.0, fs_weight=0.0):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        if batch is None:
            continue

        mel = batch.mels.to(device)        # (B, F, T)
        x = mel.permute(0, 2, 1)           # (B, T, F)
        mel_length = batch.mel_length.to(device)

        text_full = batch.text.to(device)
        text_length_full = batch.text_length.to(device)
        text_filtered, text_length_filtered = filter_batch_targets(text_full, text_length_full)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type if isinstance(device, torch.device) else (device if isinstance(device, str) else "cuda"), dtype=torch.bfloat16):
            ctc, enc_lengths = model(x, mel_length)  # ctc: (T, N, C)

            # Use encoder output lengths (true frame counts after TDNN strides/padding),
            # not raw mel_length, so CTC loss never sees padding frames.
            loss_ctc = ctc_loss_fn(
                ctc, text_filtered, enc_lengths, text_length_filtered, step_type="train"
            ) if ctc_weight > 0 else torch.zeros((), device=ctc.device)

        loss_fs = forward_sum_loss(ctc.float(), text_filtered, enc_lengths, text_length_filtered) \
            if fs_weight > 0 else torch.zeros((), device=ctc.device)
        loss_total = ctc_weight * loss_ctc + fs_weight * loss_fs
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss_total.item()

        if (i + 1) % 10 == 0:
            logger.info(
                f"[Epoch {epoch}/{num_epochs}] Iter {i+1}/{len(dataloader)} "
                f"align_loss: {loss_ctc.item():.4f} fwd_sum: {loss_fs.item():.4f}"
            )

    return running_loss / max(len(dataloader), 1)

@torch.no_grad()
def validate(model, dataloader, ctc_loss_fn, device):
    """
    1:1 match of stylish-tts validate_alignment (stage_type.py).
    Adaptations vs stylish-tts:
      - text_filtered is used instead of batch.text because the user's alphabet
        contains punctuation / non-CTC-valid tokens that must be stripped first.
      - skip guard retained: forced_align panics if mel_len < text_len.
    """
    model.eval()
    running_loss    = 0.0
    confidence_total = 0.0
    confidence_count = 0
    blank = ctc_loss_fn.blank

    for batch in dataloader:
        if batch is None:
            continue

        mel       = batch.mels.to(device)          # (B, F, T)
        x         = mel.permute(0, 2, 1)           # (B, T, F)
        mel_length = batch.mel_length.to(device)

        text_full        = batch.text.to(device)
        text_length_full = batch.text_length.to(device)
        text_filtered, text_length_filtered = filter_batch_targets(
            text_full, text_length_full
        )

        ctc, enc_lengths = model(x, mel_length)   # (T, N, C)

        # Use encoder output lengths so CTC loss and forced-align never see padding frames.
        loss = ctc_loss_fn(
            ctc, text_filtered, enc_lengths, text_length_filtered, step_type="eval"
        )
        running_loss += loss.item()

        # Confidence — 1:1 with stylish-tts validate_alignment
        logprobs = ctc.permute(1, 0, 2)            # (N, T, C)

        for i in range(mel.shape[0]):
            curr_text_len = text_length_filtered[i].item()
            curr_mel_len  = enc_lengths[i].item()
            # Safety guard: forced_align panics if sequence is longer than frames
            if curr_mel_len < curr_text_len or curr_text_len == 0:
                continue

            # torchaudio.forced_align requires log_probs.shape[1] == input_lengths,
            # so slice to the actual mel length (removes padding frames).
            _, scores = torchaudio.functional.forced_align(
                log_probs=logprobs[i, :curr_mel_len].unsqueeze(0).contiguous(),
                targets=text_filtered[i, :curr_text_len].unsqueeze(0),
                input_lengths=enc_lengths[i].unsqueeze(0),
                target_lengths=text_length_filtered[i].unsqueeze(0),
                blank=blank,
            )
            confidence_total += scores.exp().sum()
            confidence_count += scores.shape[-1]

    avg_loss = running_loss / max(len(dataloader), 1)
    avg_conf = confidence_total / max(confidence_count, 1)
    return avg_loss, avg_conf

def main():
    parser = argparse.ArgumentParser(description="Train stylish aligner CTC model")
    parser.add_argument("--data-root",   default="../my_tts_dataset_v2")
    parser.add_argument("--dump-root",   default="../tts_model_v2/dump_100")
    parser.add_argument("--train-list",  default="../my_tts_dataset_v2/train_list.txt")
    parser.add_argument("--val-list",    default="../my_tts_dataset_v2/val_list.txt")
    parser.add_argument("--output-dir",  default="stylish_aligner_outputs")
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--epochs",      type=int,   default=35)
    parser.add_argument("--ctc-weight",  type=float, default=1.0)
    parser.add_argument("--vt-rounds",   type=int,   default=0,
                        help="Viterbi-training rounds after the CTC epochs (0 = off)")
    parser.add_argument("--vt-epochs",   type=int,   default=5, help="CE epochs per Viterbi-training round")
    parser.add_argument("--vt-lr",       type=float, default=3e-4)
    parser.add_argument("--vt-min-dur",  type=int,   default=1, help="min frames per phone in the HMM realignment")
    parser.add_argument("--vt-no-optional-sil", action="store_true", help="realign with every SIL >= 1 frame")
    parser.add_argument("--init-from",   default=None, help="start from this checkpoint (e.g. a trained CTC aligner)")
    parser.add_argument("--forward-sum-weight", type=float, default=0.0,
                        help="blank-free HMM forward-sum loss weight (0 = CTC only, as before)")
    parser.add_argument(
        "--cache-all-mels-in-ram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload all mel spectrograms for this split into RAM before training.",
    )
    args = parser.parse_args()

    DATA_ROOT       = args.data_root
    DUMP_ROOT       = args.dump_root
    TRAIN_LIST_PATH = args.train_list
    VAL_LIST_PATH   = args.val_list
    OUTPUT_DIR      = args.output_dir

    BATCH_SIZE    = args.batch_size
    NUM_WORKERS   = args.num_workers
    LEARNING_RATE = args.lr
    EPOCHS        = args.epochs
    CACHE_ALL_MELS_IN_RAM = args.cache_all_mels_in_ram

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Data
    logger.info(f"Loading data from {DATA_ROOT}")
    try:
        with open(TRAIN_LIST_PATH, "r", encoding="utf-8") as f:
            train_lines = f.readlines()
        with open(VAL_LIST_PATH, "r", encoding="utf-8") as f:
            val_lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to load lists: {e}")
        return

    train_loader = build_dataloader(
        train_lines, DATA_ROOT, dump_root=DUMP_ROOT, validation=False,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        device=str(device), list_path=TRAIN_LIST_PATH,
        cache_all_mels_in_ram=CACHE_ALL_MELS_IN_RAM,
    )
    val_loader = build_dataloader(
        val_lines, DATA_ROOT, dump_root=DUMP_ROOT, validation=True,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        device=str(device), list_path=VAL_LIST_PATH,
        cache_all_mels_in_ram=CACHE_ALL_MELS_IN_RAM,
    )

    # Model — 1:1 with stylish-tts
    num_symbols = len(symbols)
    model = tdnn_blstm_ctc_model_base(n_mels=100, num_symbols=num_symbols).to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        logger.info(f"Initialised from {args.init_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Loss — 1:1 with stylish-tts TrainContext:
    #   CTCLossWithLabelPriors(prior_scaling_factor=0.3, blank=model_config.text_encoder.tokens)
    ctc_loss = CTCLossWithLabelPriors(
        prior_scaling_factor=0.3,
        blank=BLANK_TOKEN_ID,
        reduction="mean",
    ).to(device)

    best_conf = -1.0
    scaler = torch.amp.GradScaler(device, enabled=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, ctc_loss, device, epoch, EPOCHS,
            ctc_weight=args.ctc_weight, fs_weight=args.forward_sum_weight,
        )
        ctc_loss.on_train_epoch_end()

        val_loss, val_conf = validate(model, val_loader, ctc_loss, device)

        logger.info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Conf={val_conf:.4f}"
        )

        torch.save(model.state_dict(), osp.join(OUTPUT_DIR, "latest.pt"))
        if val_conf > best_conf:
            best_conf = val_conf
            torch.save(model.state_dict(), osp.join(OUTPUT_DIR, "best.pt"))
            logger.info(f"New best model saved with confidence {best_conf:.4f}")

    if args.vt_rounds > 0:
        align_loader = build_dataloader(
            train_lines, DATA_ROOT, dump_root=DUMP_ROOT, validation=True,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=str(device),
            list_path=None, cache_all_mels_in_ram=CACHE_ALL_MELS_IN_RAM,
        )
        vt_optimizer = torch.optim.Adam(model.parameters(), lr=args.vt_lr, weight_decay=1e-5)
        for r in range(args.vt_rounds):
            labels = frame_labels(model, align_loader, ctc_loss, device, use_hmm=r > 0,
                                  min_dur=args.vt_min_dur, optional_sil=not args.vt_no_optional_sil)
            logger.info(f"Viterbi training round {r + 1}/{args.vt_rounds}: {len(labels)} utterances labelled "
                        f"({'HMM' if r > 0 else 'CTC optional-SIL'} alignment)")
            for ep in range(args.vt_epochs):
                ce = train_ce_epoch(model, train_loader, vt_optimizer, labels, device)
                logger.info(f"  VT round {r + 1} epoch {ep + 1}/{args.vt_epochs}: frame CE={ce:.4f}")
            torch.save(model.state_dict(), osp.join(OUTPUT_DIR, "latest.pt"))


if __name__ == "__main__":
    main()
