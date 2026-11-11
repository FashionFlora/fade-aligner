# Fade Aligner. Derived from Stylish-TTS (https://github.com/Stylish-TTS/Stylish-TTS, MIT, Copyright (c) 2023 Aaron (Yinghao) Li).
# HMM decoding with CTC label priors follows Huang et al., ICASSP 2024 (arXiv 2406.02560). See NOTICE / LICENSE.
import os
import math
import argparse
import torch
import torchaudio
import logging
from tqdm import tqdm
from safetensors.torch import save_file

from stylish_aligner_train import (
    CTCLossWithLabelPriors,
    tdnn_blstm_ctc_model_base,
    filter_batch_targets,
    symbols,
    BLANK_TOKEN_ID,
    SIL_ID,
    SPOKEN_IDS,
    build_dataloader,
)

from alignment_hmm import nonblank_log_probs, expand, emissions, viterbi_durations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUNCTUATION_TOKENS = set(';:,.!?¡¿—–…"«»"“”')


# -----------------------------------------------------------------------------
# BATCHED K2 ALIGNMENT — 1:1 from stylish-tts align_text.py::k2_align
# Reference: stylish-tts/src/stylish_tts/train/dataprep/align_text.py
# Uses CTCLossWithLabelPriors.forced_align (k2 batched shortest-path alignment).
# prediction: (N, T, C)  — already permuted before calling.
# Returns: durations (list of Tensors), scores (Tensor [N])
# -----------------------------------------------------------------------------

def k2_align(text_filtered, mel_lengths, text_length_filtered, prediction, align_loss):
    """
    1:1 match of stylish-tts::k2_align.

    Args:
        text_filtered:        (N, S) filtered spoken token ids
        mel_lengths:          (N,)   valid frame counts
        text_length_filtered: (N,)   valid spoken token counts
        prediction:           (N, T, C) log-softmax CTC predictions
        align_loss:           CTCLossWithLabelPriors instance (provides forced_align)

    Returns:
        durations:  list of length N — each element is a 1-D LongTensor
                    [prefix_dur, tok_0_dur, ..., tok_{S-1}_dur, suffix_dur]
                    matching stylish-tts k2_align output exactly.
        scores:     Tensor [N] total-score / target_length per sample
    """
    batch_frame_labels, scores = align_loss.forced_align(
        log_probs=prediction,
        targets=text_filtered,
        input_lengths=mel_lengths,
        target_lengths=text_length_filtered,
    )

    durations = []
    all_token_indices = []  # per-sample list of token-run start frame positions
    for i, seq_frames in enumerate(batch_frame_labels):
        total_frames = mel_lengths[i].item()
        seq_frames = seq_frames[:total_frames]

        # Non-blank = any label > 0  (k2 CTC graph uses 0 as blank internally)
        token_indices = [idx for idx, label in enumerate(seq_frames) if label > 0]

        if not token_indices:
            logger.warning(
                "WARNING: no tokens found in k2 alignment, likely an untrained model."
            )
            durations.append(torch.tensor([total_frames]))
            all_token_indices.append([])
            continue

        first_idx = token_indices[0]
        last_idx = token_indices[-1]

        # Frames before the first non-blank token = prefix silence
        prefix_dur = first_idx

        # Improved duration splitting:
        # Group consecutive non-blank frames into token runs. Then create one
        # continuous partition of frames by placing boundaries at midpoints
        # between adjacent runs. This keeps every inter-token blank frame owned
        # by exactly one neighbouring token instead of dropping half of each gap
        # and later dumping the accumulated difference onto the final token.
        groups = []  # list of (label, first_idx, last_idx)
        curr_label = None
        curr_first = None
        last_nonblank = None
        in_run = False
        for pos in range(first_idx, last_idx + 1):
            lab = int(seq_frames[pos].item()) if hasattr(seq_frames[pos], 'item') else int(seq_frames[pos])
            if lab > 0:
                if not in_run:
                    curr_label = lab
                    curr_first = pos
                    in_run = True
                elif lab != curr_label:
                    groups.append((curr_label, curr_first, last_nonblank))
                    curr_label = lab
                    curr_first = pos
                last_nonblank = pos
            elif in_run:
                groups.append((curr_label, curr_first, last_nonblank))
                curr_label = None
                curr_first = None
                in_run = False

        if in_run:
            groups.append((curr_label, curr_first, last_nonblank))

        token_durs = []
        if len(groups) == 0:
            # Fallback: everything is blanks (should be handled earlier)
            token_durs.append(total_frames)
        else:
            boundaries = [groups[0][1]]
            for j in range(len(groups) - 1):
                prev_last = groups[j][2]
                next_first = groups[j + 1][1]
                boundaries.append((prev_last + next_first + 1) // 2)
            boundaries.append(total_frames)

            for j in range(len(groups)):
                dur = boundaries[j + 1] - boundaries[j]
                token_durs.append(max(1, int(dur)))

        # Keep suffix as 0 (last token already absorbs trailing frames)
        suffix_dur = 0

        # [prefix, tok_0, ..., tok_{S-1}, suffix]  — 1:1 with stylish-tts
        full_durs = [int(prefix_dur)] + token_durs + [int(suffix_dur)]
        durations.append(torch.tensor(full_durs, dtype=torch.long))
        all_token_indices.append([g_first for _, g_first, _ in groups])

    return durations, scores, all_token_indices


# -----------------------------------------------------------------------------
# BATCH ALIGNMENT DRIVER — k2 batched, with remap back to full text positions
# -----------------------------------------------------------------------------

def align_batch(model, batch, device, align_loss):
    """
    Run batched k2 forced alignment on one batch.
    Maps k2 durations back to full text positions (including punctuation).

    Returns a list of (full_dur, sample_score) per sample, or None on failure.
      full_dur:     (full_text_len,) LongTensor — frame durations per token.
                    Spoken tokens get durations from k2 alignment.
                    Non-spoken tokens (punctuation) get duration = 0.
                    First spoken token ($ boundary) absorbs prefix silence.
                    Last spoken token ($ boundary) absorbs suffix silence.
      sample_score: scalar float — exp(total_score / target_len).
    """
    mel         = batch.mels.to(device)          # (B, F, T)
    x           = mel.permute(0, 2, 1)           # (B, T, F)
    mel_length  = batch.mel_length.to(device)    # (B,)

    text_full        = batch.text.to(device)         # (B, L_full)
    text_length_full = batch.text_length.to(device)  # (B,)
    text_filtered, text_length_filtered = filter_batch_targets(
        text_full, text_length_full
    )

    # CTC forward:  ctc is (T, N, C);  permute to (N, T, C) for k2
    ctc, enc_lengths = model(x, mel_length)
    log_probs = ctc.permute(1, 0, 2).contiguous()   # (N, T, C)

    # Batched k2 forced alignment — 1:1 with stylish-tts calculate_alignment_batched
    try:
        durations, scores, token_indices_list = k2_align(
            text_filtered, enc_lengths, text_length_filtered, log_probs, align_loss
        )
    except Exception as e:
        logger.warning(f"k2_align failed for batch: {e}")
        return [None] * len(batch.names)

    results = []
    for i in range(len(batch.names)):
        curr_text_len  = text_length_filtered[i].item()  # number of spoken tokens
        full_text_len  = text_length_full[i].item()      # includes punctuation
        curr_mel_len   = enc_lengths[i].item()

        if curr_text_len == 0 or i >= len(durations):
            results.append(None)
            continue

        k2_durs = durations[i]  # 1-D LongTensor [prefix, tok0, ..., tok_{S-1}, suffix]
        
        # Compute sample_score using exp of per-frame average
        if curr_mel_len < curr_text_len:
            sample_score = 0.0
        else:
            _, ta_scores = torchaudio.functional.forced_align(
                log_probs=log_probs[i, :curr_mel_len].unsqueeze(0).contiguous(),
                targets=text_filtered[i, :curr_text_len].unsqueeze(0),
                input_lengths=enc_lengths[i].unsqueeze(0),
                target_lengths=text_length_filtered[i].unsqueeze(0),
                blank=BLANK_TOKEN_ID,
            )
            sample_score = ta_scores.exp().mean().item()

        # Extract prefix, inner_durs, suffix from k2_align output
        # k2_durs = [prefix_dur, tok0_dur, tok1_dur, ..., tok_{S-1}_dur, suffix_dur]
        prefix_dur = k2_durs[0].item()
        suffix_dur = k2_durs[-1].item()
        inner_durs = k2_durs[1:-1]  # middle elements

        # Map back to full text: spoken tokens get durations, non-spoken get 0
        full_dur = torch.zeros(full_text_len, dtype=torch.long, device=device)
        full_text_tokens = text_full[i, :full_text_len].cpu().tolist()

        # Identify spoken token positions in full text
        spoken_positions = [
            pos for pos, tid in enumerate(full_text_tokens) if tid in SPOKEN_IDS
        ]

        # Handle mismatch between k2 output and spoken positions
        # K2 can produce more/fewer segments due to CTC collapse behavior
        exact_spoken_mapping = len(inner_durs) == len(spoken_positions)
        if not exact_spoken_mapping:
            logger.warning(
                f"Duration-position mismatch for {batch.names[i]}: "
                f"inner_durs={len(inner_durs)} vs spoken_positions={len(spoken_positions)} "
                f"prefix={prefix_dur} suffix={suffix_dur} inner_sum={sum(d.item() for d in inner_durs)}"
            )
            # Distribute durations: if mismatch, fold excess into last token or pad with 1s
            if len(inner_durs) > len(spoken_positions):
                # More durations than positions: fold overflow into last position
                for k in range(len(spoken_positions) - 1):
                    full_dur[spoken_positions[k]] = inner_durs[k].item()
                # Sum remaining durations into last position
                full_dur[spoken_positions[-1]] = sum(d.item() for d in inner_durs[len(spoken_positions)-1:])
            else:
                # Fewer durations than positions: assign what we have
                for k in range(len(inner_durs)):
                    full_dur[spoken_positions[k]] = inner_durs[k].item()
                # Don't pad remaining positions yet - we'll fix it below based on total mel frames
        else:
            # Perfect match: assign directly
            for k, pos in enumerate(spoken_positions):
                full_dur[pos] = inner_durs[k].item()

        # First spoken token absorbs prefix silence
        full_dur[spoken_positions[0]] += prefix_dur
        
        # Last spoken token absorbs suffix silence
        full_dur[spoken_positions[-1]] += suffix_dur

        # Internal punctuation pauses: CTC's minimum path often emits blanks between
        # the final phoneme before punctuation and the following SIL token. The raw
        # duration extraction assigns those blanks to the preceding phoneme. If the
        # model starts preferring SIL inside that gap, move the tail to the SIL token.
        t_indices = token_indices_list[i] if i < len(token_indices_list) else []
        if exact_spoken_mapping and len(t_indices) >= curr_text_len:
            for k in range(len(spoken_positions) - 1):
                prev_pos = spoken_positions[k]
                sil_pos = spoken_positions[k + 1]

                if full_text_tokens[prev_pos] == SIL_ID:
                    continue
                if full_text_tokens[sil_pos] != SIL_ID:
                    continue

                has_punctuation_between = any(
                    full_text_tokens[pos] < len(symbols)
                    and symbols[full_text_tokens[pos]] in PUNCTUATION_TOKENS
                    for pos in range(prev_pos + 1, sil_pos)
                )
                if not has_punctuation_between:
                    continue

                prev_frame = t_indices[k]
                sil_frame = t_indices[k + 1]
                if sil_frame <= prev_frame:
                    continue

                prev_label = text_filtered[i, k].item()
                boundary = sil_frame
                region = log_probs[i, prev_frame:sil_frame]
                for t in range(region.shape[0]):
                    if region[t, SIL_ID] > region[t, prev_label]:
                        boundary = prev_frame + t
                        break

                original_prev_dur = int(full_dur[prev_pos].item())
                new_prev_dur = max(1, min(original_prev_dur, boundary - prev_frame))
                moved_to_sil = original_prev_dur - new_prev_dur
                if moved_to_sil > 0:
                    full_dur[prev_pos] = new_prev_dur
                    full_dur[sil_pos] += moved_to_sil

        # Force trailing SIL to absorb all tail frames after last non-SIL spoken token.
        # This prevents large pre-final blank regions from staying on the last letter.
        trailing_sil_pos = None
        if spoken_positions and full_text_tokens[spoken_positions[-1]] == SIL_ID:
            trailing_sil_pos = spoken_positions[-1]

        if trailing_sil_pos is not None:
            last_content_pos = None
            for pos in reversed(spoken_positions[:-1]):
                if full_text_tokens[pos] != SIL_ID:
                    last_content_pos = pos
                    break

            if last_content_pos is not None:
                for pos in spoken_positions:
                    if pos > last_content_pos and pos != trailing_sil_pos:
                        full_dur[pos] = 0

                # Determine the true phN / trailing-SIL boundary using log_probs.
                # CTC minimum-path causes all blanks between phN's label frame and
                # SIL_end's label frame to be attributed to phN.  The true boundary
                # is the first frame in that ambiguous region where p(SIL) > p(phN).
                if len(t_indices) >= 2:
                    ph_n_frame   = t_indices[-2]   # CTC frame where phN label appears
                    sil_end_frame = t_indices[-1]  # CTC frame where SIL_end label appears
                    ph_n_label   = text_filtered[i, curr_text_len - 2].item()

                    # Scan the ambiguous region for the first frame where
                    # the model prefers SIL over phN — that is the natural boundary.
                    boundary = sil_end_frame  # conservative default: keep CTC's split
                    if sil_end_frame > ph_n_frame:
                        region = log_probs[i, ph_n_frame:sil_end_frame]  # (gap, C)
                        for t in range(region.shape[0]):
                            if region[t, SIL_ID] > region[t, ph_n_label]:
                                boundary = ph_n_frame + t
                                break

                    frames_before_ph_n = int(full_dur[:last_content_pos].sum().item())
                    ph_n_dur = max(1, boundary - ph_n_frame)
                    full_dur[last_content_pos]  = ph_n_dur
                    full_dur[trailing_sil_pos]  = max(0, curr_mel_len - frames_before_ph_n - ph_n_dur)
                else:
                    # Fallback (no token_indices info): redistribute remaining frames
                    tail_frames = curr_mel_len - int(full_dur[:last_content_pos + 1].sum().item())
                    full_dur[trailing_sil_pos] = max(0, tail_frames)

        # Ensure total duration matches mel frames
        total_assigned = full_dur.sum().item()
        if total_assigned != curr_mel_len:
            diff = curr_mel_len - total_assigned
            logger.debug(
                f"Adjusting total duration for {batch.names[i]}: "
                f"assigned={total_assigned}, expected={curr_mel_len}, diff={diff}"
            )
            # Adjust the last spoken token to make up the difference
            full_dur[spoken_positions[-1]] += diff

        results.append((full_dur.cpu(), sample_score))

    return results


def estimate_hmm_log_priors(model, loader, device):
    """Class priors for the HMM decode: log of the average non-blank frame
    posterior on the data being aligned, floored at -12."""
    total, n = None, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            ctc, enc_lengths = model(batch.mels.to(device).permute(0, 2, 1), batch.mel_length.to(device))
            lp = nonblank_log_probs(ctc.permute(1, 0, 2))
            frames = torch.cat([lp[i, :int(l)] for i, l in enumerate(enc_lengths)], 0)
            s = torch.logsumexp(frames, 0)
            total = s if total is None else torch.logaddexp(total, s)
            n += frames.shape[0]
    return (total - math.log(n)).clamp(min=-12.0)


def align_batch_hmm(model, batch, device, optional_sil=True, min_dur=1, min_pause=1, log_priors=None, prior_scale=0.0):
    """
    Blank-free HMM Viterbi on the aligner's non-blank posteriors: every frame
    belongs to one token, inner word-boundary SILs may take 0 frames, other
    tokens take >= min_dur frames; with min_pause > 1 an inner SIL takes 0 or
    >= min_pause frames. Same output contract as align_batch.
    With log_priors, emissions are hybrid-HMM scaled likelihoods
    posterior / prior^prior_scale; the score still uses the posteriors.
    """
    mel = batch.mels.to(device)
    mel_length = batch.mel_length.to(device)
    text_full = batch.text.to(device)
    text_length_full = batch.text_length.to(device)
    text_filtered, text_length_filtered = filter_batch_targets(text_full, text_length_full)
    ctc, enc_lengths = model(mel.permute(0, 2, 1), mel_length)
    lp = nonblank_log_probs(ctc.permute(1, 0, 2))
    pos = torch.arange(text_filtered.shape[1], device=device)[None, :]
    skippable = (text_filtered == SIL_ID) & (pos > 0) & (pos < text_length_filtered[:, None] - 1)
    if not optional_sil:
        skippable = torch.zeros_like(skippable)
    st, st_len, st_skip, st_tok = expand(text_filtered.cpu(), text_length_filtered.cpu(), skippable.cpu(),
                                         min_dur=min_dur, single_state={SIL_ID}, skip_len=min_pause)
    lp_emit = lp
    if log_priors is not None and prior_scale > 0:
        lp_emit = lp - prior_scale * log_priors.to(lp.device)
    emit = emissions(lp_emit, st.to(lp.device), st_len)
    st_dur, feasible = viterbi_durations(emit, enc_lengths.long(), st_len, st_skip, skip_len=min_pause)
    st_dur = st_dur.cpu()
    results = []
    for i in range(len(batch.names)):
        L = int(text_length_filtered[i]); full_len = int(text_length_full[i])
        if L == 0 or not bool(feasible[i]):
            results.append(None)
            continue
        tok_dur = torch.zeros(L, dtype=torch.long)
        n = int(st_len[i])
        tok_dur.scatter_add_(0, st_tok[i, :n], st_dur[i, :n])
        full_tokens = text_full[i, :full_len].cpu().tolist()
        spoken_positions = [p for p, t in enumerate(full_tokens) if t in SPOKEN_IDS]
        full_dur = torch.zeros(full_len, dtype=torch.long)
        full_dur[spoken_positions] = tok_dur
        # score: mean per-frame log-prob of the chosen path (comparable across utterances)
        T = int(enc_lengths[i])
        tok_of_frame = torch.repeat_interleave(torch.arange(L), tok_dur)
        path_lp = lp[i, torch.arange(T), text_filtered[i, tok_of_frame.to(lp.device)]]
        results.append((full_dur, float(path_lp.mean().exp())))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run k2 forced alignment with stylish aligner")
    parser.add_argument("--data-root",   default="../my_tts_dataset_v2")
    parser.add_argument("--dump-root",   default="../tts_model_v2/dump_100")
    parser.add_argument("--train-list",  default="../my_tts_dataset_v2/train_list.txt")
    parser.add_argument("--val-list",    default="../my_tts_dataset_v2/val_list.txt")
    parser.add_argument("--model-path",  default="stylish_aligner_outputs/latest.pt")
    parser.add_argument("--output-dir",  default=".")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cache-all-mels-in-ram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload all mel spectrograms for this split into RAM before alignment.",
    )
    parser.add_argument("--decode", choices=["ctc", "hmm"], default="ctc",
                        help="ctc: k2 CTC alignment + midpoint rule (base); hmm: blank-free Viterbi")
    parser.add_argument("--no-optional-sil", action="store_true", help="hmm: every SIL takes >= 1 frame")
    parser.add_argument("--min-dur", type=int, default=1, help="hmm: minimum frames per phone")
    parser.add_argument("--min-pause", type=int, default=1,
                        help="hmm: a word-boundary SIL takes 0 or at least this many frames")
    parser.add_argument("--hmm-prior-scale", type=float, default=0.0,
                        help="hmm: divide posteriors by class priors^scale (priors estimated "
                             "from the data being aligned)")
    args = parser.parse_args()

    DATA_ROOT       = args.data_root
    DUMP_ROOT       = args.dump_root
    TRAIN_LIST_PATH = args.train_list
    VAL_LIST_PATH   = args.val_list
    MODEL_PATH      = args.model_path
    OUTPUT_DIR      = args.output_dir

    BATCH_SIZE  = args.batch_size
    NUM_WORKERS = args.num_workers
    CACHE_ALL_MELS_IN_RAM = args.cache_all_mels_in_ram
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Run training first.")
        return

    # --- Load model ---
    num_symbols = len(symbols)
    model = tdnn_blstm_ctc_model_base(n_mels=100, num_symbols=num_symbols).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    logger.info(f"Model loaded from {MODEL_PATH}")

    # --- Shared CTCLossWithLabelPriors instance (provides forced_align) ---
    # 1:1 with stylish-tts TrainContext: CTCLossWithLabelPriors(prior_scaling_factor=0.3, blank=...)
    align_loss = CTCLossWithLabelPriors(
        prior_scaling_factor=0.3,
        blank=BLANK_TOKEN_ID,
        reduction="mean",
    ).to(DEVICE)

    for list_path, split_name in [(VAL_LIST_PATH, "val"), (TRAIN_LIST_PATH, "train")]:
        if not os.path.exists(list_path):
            logger.warning(f"{split_name} list not found at {list_path}. Skipping.")
            continue

        logger.info(f"Processing {split_name}: {list_path}")

        with open(list_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) == 0:
            logger.info(f"Skipping {split_name} because {list_path} is empty.")
            continue

        loader = build_dataloader(
            lines,
            DATA_ROOT,
            dump_root=DUMP_ROOT,
            validation=True,          # shuffle=False; no augmentation
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=str(DEVICE),
            list_path=list_path,
            cache_all_mels_in_ram=CACHE_ALL_MELS_IN_RAM,
        )

        safetensors_dict = {}          # name -> duration Tensor
        scores_map       = {}          # name -> float score  (like stylish-tts scores_*.txt)

        log_priors = None
        if args.decode == "hmm" and args.hmm_prior_scale > 0:
            log_priors = estimate_hmm_log_priors(model, loader, DEVICE)
            logger.info(f"Estimated class priors on {split_name}; HMM decode with prior scale {args.hmm_prior_scale}")

        logger.info(f"Starting k2 alignment on {len(loader.dataset)} files…")

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Aligning {split_name}"):
                if batch is None:
                    continue

                if args.decode == "hmm":
                    batch_results = align_batch_hmm(model, batch, DEVICE, optional_sil=not args.no_optional_sil,
                                                    min_dur=args.min_dur, min_pause=args.min_pause,
                                                    log_priors=log_priors, prior_scale=args.hmm_prior_scale)
                else:
                    batch_results = align_batch(model, batch, DEVICE, align_loss)

                for name, res in zip(batch.names, batch_results):
                    if res is not None:
                        full_dur, score = res
                        safetensors_dict[name] = full_dur
                        scores_map[name]       = score

        # Save alignments — 1:1 with stylish-tts save_file call
        out_path = os.path.join(OUTPUT_DIR, f"alignments_{split_name}.safetensors")
        if os.path.exists(out_path):
            os.unlink(out_path)
        save_file(safetensors_dict, out_path)
        logger.info(f"Saved {len(safetensors_dict)} alignment tensors → {out_path}")

        # Save scores — 1:1 with stylish-tts scores_*.txt format
        scores_path = os.path.join(OUTPUT_DIR, f"scores_{split_name}.txt")
        with open(scores_path, "w", encoding="utf-8") as f:
            for name, score in scores_map.items():
                f.write(f"{score} {name}\n")
        logger.info(f"Saved scores → {scores_path}")


if __name__ == "__main__":
    main()
