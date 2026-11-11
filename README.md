<h1 align="center">Fade Aligner</h1>

<p align="center"><b>Fast phoneme-level forced alignment: phoneme durations and boundaries for any espeak-ng language.</b></p>

<p align="center">
  <a href="https://github.com/FashionFlora/fade-aligner"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-fade--aligner-black?logo=github"></a>
  <a href="https://huggingface.co/FashionFlora/fade-aligner"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow"></a>
  <a href="https://fashionflora.github.io/fade-aligner/"><img alt="Project page" src="https://img.shields.io/badge/Project-page-blue"></a>
  <a href="https://arxiv.org/abs/2406.02560"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2406.02560-b31b1b"></a>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

**Built on [Stylish-TTS](https://github.com/Stylish-TTS/Stylish-TTS)** (MIT, Copyright (c) 2023 Aaron (Yinghao) Li): the
model, CTC loss and k2 alignment driver come from its aligner, which in turn builds on
[StyleTTS 2](https://github.com/yl4579/StyleTTS2). Label-prior decoding follows
[Huang et al., ICASSP 2024](https://arxiv.org/abs/2406.02560).

## Overview

Fade Aligner takes audio plus its transcript and returns, for every phoneme, a start and an end time. The transcript is
phonemized with espeak-ng, the acoustic model is a small TDNN-BLSTM trained with CTC on 100-bin mel spectrograms, and the
alignment is then decoded by an HMM Viterbi pass.

Training is a three-step recipe:

1. **CTC with label priors** (35 epochs by default) trains the frame posteriors.
2. **Viterbi training rounds** re-align the training set with the model itself and train it again with frame-level
   cross-entropy on those alignments, which sharpens the boundaries.
3. **HMM decoding** (`--decode hmm`): a blank-free monotonic Viterbi search with a minimum phone length, optional
   word-boundary pauses and posteriors divided by class priors (`--hmm-prior-scale`).

## Highlights

- Phoneme-level boundaries from an espeak-ng phonemization; works for every language espeak-ng supports.
- GPU Viterbi for decoding and for the training rounds; mel spectrograms are computed on the GPU
  (`ALIGN_MELS_ON_GPU=1`, default) from a compact FLAC store (`audio.parquet`), so there are no mel files on disk.
- One model can cover several languages: the phoneme symbol table is shared.
- Output as JSON per clip and as Praat TextGrid (phones and words tiers).

## Quick start

```bash
# 1. install (Python 3.10+); k2 needs a wheel that matches torch/CUDA, see requirements.txt
sudo apt install espeak-ng
pip install -r requirements.txt

# 2. build the dataset (repeat with --append and another --lang to mix languages in one model)
python tools/prepare_data.py --format libritts --input LibriTTS_R/train-clean-100 --lang en-us --out data

# 3. train and align (the demo model used exactly these flags)
cd aligner
python stylish_aligner_train.py --data-root ../data --dump-root ../data/audio.parquet \
  --train-list ../data/train_list.txt --val-list ../data/val_list.txt --output-dir ../data/model \
  --epochs 35 --vt-rounds 2 --vt-epochs 5 --vt-lr 3e-4 --vt-min-dur 2
python stylish_aligner_align.py --data-root ../data --dump-root ../data/audio.parquet \
  --train-list ../data/train_list.txt --val-list ../data/val_list.txt \
  --model-path ../data/model/latest.pt --output-dir ../data/align \
  --decode hmm --min-dur 3 --hmm-prior-scale 1.0 --min-pause 5
cd ..

# 4. export JSON + TextGrid
python tools/export_alignments.py --align-dir data/align --split val --list data/val_list.txt \
  --audio data/audio.parquet --out export --n 10 --textgrid --save-audio
```

To align your own recordings with an existing model in one command (folder of `name.wav` + `name.txt`):

```bash
python tools/align_folder.py --input my_clips --model model/latest.pt --lang en-us --out out
```

## Output format

One JSON per clip:

```json
{"key": "...", "text": "...", "ipa": "...", "sr": 44100, "duration_s": 4.21, "score": 0.93,
 "tokens": [["<b>", 0.0, 0.0], ["_", 0.0, 0.18], ["ð", 0.18, 0.23], ["ə", 0.23, 0.27], "..."]}
```

- `tokens` are `[symbol, start_s, end_s]` in text order. `_` is the space / silence token (including the leading and
  trailing silence), `<b>` are two zero-width boundary markers. Punctuation and stress marks take zero frames.
- Frames are 10 ms. The aligner pads the audio by 4 frames per side before the mels; the export subtracts this.
- The TextGrid has a `phones` tier and a `words` tier (tokens between silences, labelled with the IPA word).

## Data format

`train_list.txt` / `val_list.txt`: `relative/key.flac|IPA phonemes|lang_id|speaker_id|0|original text`.
`audio.parquet`: columns `key` (list key without `.flac`) and `audio` (44.1 kHz mono FLAC bytes).
`lang_id` and `speaker_id` are read and ignored by the aligner. `tools/prepare_data.py merge OUT A B` concatenates
finished datasets.

## CLI reference

`tools/prepare_data.py`: `--input`, `--format {folder,libritts,mls}`, `--lang` (espeak-ng voice), `--lang-id`, `--out`,
`--append`, `--max-hours`, `--max-clips`, `--min-s` / `--max-s` (clip length, default 1-20 s), `--val-frac` (0.02),
`--seed`, `--workers`, `--prefix`.

`aligner/stylish_aligner_train.py`: `--data-root --dump-root --train-list --val-list --output-dir --batch-size (64)
--num-workers --lr (1e-4) --epochs (35) --ctc-weight --vt-rounds (0) --vt-epochs (5) --vt-lr (3e-4) --vt-min-dur (1)
--vt-no-optional-sil --init-from --forward-sum-weight --cache-all-mels-in-ram`.

`aligner/stylish_aligner_align.py`: `--data-root --dump-root --train-list --val-list --model-path --output-dir --batch-size
(16) --num-workers --decode {ctc,hmm} --no-optional-sil --min-dur (1) --min-pause (1) --hmm-prior-scale (0.0)`. Writes
`alignments_<split>.safetensors` (frame counts per token) and `scores_<split>.txt`.

`tools/export_alignments.py`: `--align-dir --split --list --audio --out --keys --n --textgrid --save-audio`.

Environment: `ALIGN_MELS_ON_GPU` (1 = keep mels on the GPU), `ALIGN_DECODE_THREADS` (FLAC decoding threads, 4).

## Languages

Any language with an espeak-ng voice. Pass the voice name to `--lang` (`en-us`, `pl`, `de`, `fr-fr`, `es`, ...). The
phoneme table covers the IPA symbols espeak-ng emits for common languages; unknown symbols are logged and skipped.

## Demo model

An English model (5.0 M parameters, a 20 MB file) trained on LibriTTS-R train-clean-100 (all clips of 1-20 s, 50 h, CC BY 4.0) with the flags
shown in the quick start. Weights: https://huggingface.co/FashionFlora/fade-aligner (`model/latest.pt`, a plain PyTorch state dict).

- Interactive viewer with 12 aligned validation clips (click any phoneme to hear exactly that segment):
  https://fashionflora.github.io/fade-aligner/
- `examples/` holds the JSON and Praat TextGrid of those 12 clips.
- Mean alignment confidence on the training clips is 0.55 and 1.2 % of the clips score below 0.3 (`scores_train.txt`).
  The score is a confidence value of the alignment, not an accuracy. We have not benchmarked boundary errors against
  Montreal Forced Aligner or published a boundary-error number; for the label-prior method itself see the
  [paper](https://arxiv.org/abs/2406.02560). Please check the alignments of your own data before relying on them.

Other languages work the same way: build a dataset with `--lang <espeak-ng voice>` and train your own model.

## Limitations

- Time resolution is one mel frame, 10 ms, and a phone is at least 3 frames (30 ms) with the default `--min-dur 3`.
- The internal sample rate is 44.1 kHz. LibriTTS-R (24 kHz) is upsampled, so the demo model has seen no
  energy above 12 kHz; on full-band recordings it is untested. The same holds for other lower-rate corpora.
- The demo model is English only. A model is trained for its language set; use one trained on the language you align, or train your own.
- Quality depends on the transcripts: wrong, missing or extra words, and phonemization errors, shift boundaries.
- Clips are meant to be 1-20 s long; cut long recordings into sentences first.

## Credits and licenses

Fade Aligner is MIT-licensed (see `LICENSE`, which also keeps the upstream Stylish-TTS notice; third-party components
are listed in `NOTICE`). Thanks to [Stylish-TTS](https://github.com/Stylish-TTS/Stylish-TTS),
[StyleTTS 2](https://github.com/yl4579/StyleTTS2), [k2](https://github.com/k2-fsa/k2),
[espeak-ng](https://github.com/espeak-ng/espeak-ng) (GPL-3.0, used as an external program, installed separately),
and the authors of LibriTTS-R (CC BY 4.0, attribution in `NOTICE`).

Project page: https://fashionflora.github.io/fade-aligner/ - Model: https://huggingface.co/FashionFlora/fade-aligner - Paper
(label priors): https://arxiv.org/abs/2406.02560
