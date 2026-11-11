# Fade Aligner. Derived from Stylish-TTS (https://github.com/Stylish-TTS/Stylish-TTS, MIT, Copyright (c) 2023 Aaron (Yinghao) Li).
# HMM decoding with CTC label priors follows Huang et al., ICASSP 2024 (arXiv 2406.02560). See NOTICE / LICENSE.
# meldataset_preparation.py
# Unified constants and preprocess; consistent names for alignment keys.

import os.path as osp
import logging
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchaudio
import soundfile as sf
import yaml
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import os

_k2_blank = "<"  # dummy token at ID 0 — k2 uses this slot for blank/epsilon internally;
_pad = "$"
_punctuation = ';:,.!?¡¿—–…"«»"“”'
_sillence_token = " "
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǃᵻ"
_punctuation_ipa = "̃ˈˌːˑʼ'̩'|^‖ʴʰʱʲʷˠˤ˞↓↑→↗↘'̪ "
_espeak_multichars = [
    # 3/4-char nasal + length combinations.
    "ɔ̃ː", "ɛ̃ː", "ãː", "œ̃ː", "ẽː",
    # 2-char long vowels.
    "aː", "eː", "iː", "oː", "uː", "yː", "øː", "œː", "ɔː", "ɛː", "ɑː", "ɜː",
    "aˑ", "eˑ", "iˑ", "oˑ", "uˑ",
    # 2-char nasal vowels.
    "ã", "ɛ̃", "ɔ̃", "œ̃", "ĩ", "ũ", "ẽ", 
    # 2-char palatalized consonants.
    "pʲ", "bʲ", "mʲ", "fʲ", "vʲ", "tʲ", "dʲ", "sʲ", "zʲ", "nʲ", "lʲ", "rʲ", "kʲ", "gʲ", "xʲ", "ɭʲ", "ɲʲ", "ɕʲ", "ʃʲ",
    # 2-char aspirated consonants.
    "pʰ", "tʰ", "kʰ", "sʰ", "ʃʰ",

    "l̩", "n̩", "m̩",
    # appended later (ids of the tokens above unchanged), for more espeak-ng languages: nasal vowels that would
    # otherwise be split into vowel + non-spoken tilde (fr ɑ̃, pt ɐ̃ ʊ̃ õ), palatals with the IPA ɡ (ru / pl ɡʲ) and ʒʲ,
    # and the Italian geminates (long consonant = one token instead of consonant + non-spoken ː)
    "ɑ̃", "ɐ̃", "ʊ̃", "o\u0303", "ɡʲ", "ʒʲ",
    "tː", "kː", "pː", "bː", "dː", "ɡː", "sː", "zː", "ʃː", "ʒː", "lː", "mː", "nː", "rː", "ɲː", "ʎː", "ʦː", "ʧː", "ʤː",
]
symbols = [_k2_blank, _pad] + list(_punctuation) + list(_letters) + [_sillence_token] + list(_letters_ipa) + list(_punctuation_ipa) + _espeak_multichars + ["-"]
# "-" (appended later): espeak's French clitic / liaison mark ("de-z elɛksjˈɔ̃"), not spoken (duration 0 like
# punctuation)
dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

# --- Define sets for filtering ---
# Letters, IPA, and Silence (' ') are alignable spoken tokens.
# _k2_blank (ID 0) is never in any transcript — excluded automatically.
# Punctuation (';:,.!?...') and IPA punctuation are excluded (duration = 0).
SPOKEN_CHARS = list(_letters) + list(_letters_ipa) + [_sillence_token] + _espeak_multichars
SPOKEN_IDS = {dicts[c] for c in SPOKEN_CHARS if c in dicts}
# ------------------------------------------

#print(len(symbols), "symbols:", symbols)
PAD_ID = dicts[_k2_blank] 
SIL_ID = dicts[_sillence_token]
BLANK_TOKEN_ID = 0  # _k2_blank ("<") is at index 0 — reserved for CTC blank
SAMPLE_RATE = 44100
HOP_LENGTH = 441
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1764,
    "hop_length": HOP_LENGTH,
}
MEL_PARAMS = {"n_mels": 100}




class TextCleaner:
    def __init__(self, dummy: Optional[object] = None):
        self.word_index_dictionary = dicts
        self._unknown_chars_logged = set()
        # Longest-match-first so multichar tokens are consumed before single chars.
        self._multichars = sorted(
            [tok for tok in _espeak_multichars if tok in self.word_index_dictionary],
            key=len,
            reverse=True,
        )

    def __call__(self, text: str):
        indexes = []
        i = 0
        while i < len(text):
            matched = False
            for tok in self._multichars:
                if text.startswith(tok, i):
                    indexes.append(self.word_index_dictionary[tok])
                    i += len(tok)
                    matched = True
                    break

            if matched:
                continue

            ch = text[i]
            if ch in self.word_index_dictionary:
                indexes.append(self.word_index_dictionary[ch])
            else:
                if ch not in self._unknown_chars_logged:
                    logger.warning(
                        f"TextCleaner: unknown symbol {repr(ch)} (U+{ord(ch):04X}) — this warning is shown once per symbol"
                    )
                    self._unknown_chars_logged.add(ch)
            i += 1
        return indexes


# --- mels from an audio store (dump_root = *.parquet with columns key, audio: 44.1 kHz FLAC bytes), computed at start
# (no mel files on disk), as calculate_mel_energy_pitch_2.py without its lowpass: 1764 samples of padding each side,
# STFT n_fft 2048 / win 1764 / hop 441 (center, reflect), magnitude sqrt(re^2 + im^2 + 1e-6), librosa mel basis 100
# bins 0-16 kHz, log clamp 1e-5, length a multiple of 4.
_MELS = {}  # store path -> {key: mel fp16 [100, T]}, filled once per process
# keep the mels on the GPU (fp16, ~4 GB per 50 h): batches are padded there, no CPU copy per step
_MELS_ON_GPU = torch.cuda.is_available() and os.environ.get("ALIGN_MELS_ON_GPU", "1") == "1"


def _mel_batch(waves, dev):
    import librosa
    basis = torch.from_numpy(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=2048, n_mels=100, fmin=0, fmax=16000)).float().to(dev)
    hann = torch.hann_window(1764).to(dev)
    out = []
    for x in waves:
        x = np.concatenate([np.zeros(1764, np.float32), x, np.zeros(1764, np.float32)])
        with torch.no_grad():
            z = torch.stft(torch.from_numpy(x).to(dev)[None], n_fft=2048, hop_length=HOP_LENGTH, win_length=1764, window=hann,
                           center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
            mel = torch.log(torch.clamp(basis @ torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-6), min=1e-5))[0]
        m = mel[:, :mel.shape[1] - mel.shape[1] % 4].half()
        out.append(m.clone() if _MELS_ON_GPU else m.cpu())
    return out


def _load_store(store_path: str):
    """every clip of the store: FLAC decoded by a thread pool, mel on the GPU (CPU if none), kept in RAM as fp16; the
    audio bytes are dropped afterwards"""
    if store_path in _MELS:
        return
    import io, time
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    pf = pq.ParquetFile(store_path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def decode(b):
        x, sr = sf.read(io.BytesIO(b), dtype="float32")
        assert sr == SAMPLE_RATE, sr
        return x

    mels = {}
    with ThreadPoolExecutor(int(os.environ.get("ALIGN_DECODE_THREADS", "4"))) as ex:
        for rb in pf.iter_batches(batch_size=512, columns=["key", "audio"]):  # streamed: the store is never fully in RAM
            keys, blobs = rb.column("key").to_pylist(), rb.column("audio").to_pylist()
            ws = list(ex.map(decode, blobs))
            for key, m in zip(keys, _mel_batch(ws, dev)):
                mels[key] = m
            del blobs, ws
    _MELS[store_path] = mels
    import ctypes, gc, pyarrow as pa
    gc.collect(); pa.default_memory_pool().release_unused()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # give the decoded audio back to the OS (else ~12 GB RSS per job)
    except OSError:
        pass
    logger.info(f"mels of {len(mels)} clips computed on {dev} in {time.time() - t0:.0f} s ({store_path})")


def _mel_from_store(store_path: str, wave_path: str):
    _load_store(store_path)
    mel = _MELS[store_path][osp.splitext(wave_path)[0]]  # fp16 view (the RAM cache keeps no float copy)
    return mel[:, : (mel.shape[1] - mel.shape[1] % 2)]


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list_lines: List[str],
        root_path: str,
        dump_root: Optional[str] = None,  # <--- NEW ARGUMENT
        sr: int = SAMPLE_RATE,
        validation: bool = False,
        use_basename: bool = False,
        return_wave: bool = False,
        cache_all_mels_in_ram: bool = False,
        min_mel_text_ratio: float = 1.5,
        **kwargs,
    ):
        rows = [l.strip().split("|", 5) for l in data_list_lines]
        self.data_list = []
        for row in rows:
            if len(row) < 6:
                row += [" "] * (6 - len(row)) 
            self.data_list.append(tuple(row))

        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.root_path = root_path
        self.dump_root = dump_root # <--- Store dump root
        self.validation = validation
        self.use_basename = use_basename
        self.return_wave = return_wave
        self.cache_all_mels_in_ram = cache_all_mels_in_ram
        self.min_mel_text_ratio = min_mel_text_ratio
        self.cached_samples = None

        if dump_root is not None and dump_root.endswith(".parquet") and self.data_list:
            _mel_from_store(dump_root, self.data_list[0][0])  # load the audio store before the workers fork (shared pages)
        if self.cache_all_mels_in_ram:
            self.cached_samples = []
            skipped = 0
            for idx in range(len(self.data_list)):
                sample = self._build_sample(idx, enforce_ratio_filter=True)
                if sample is None:
                    skipped += 1
                    continue
                self.cached_samples.append(sample)
            logger.info(
                f"RAM mel cache enabled ({'val' if self.validation else 'train'}): "
                f"loaded {len(self.cached_samples)} samples, skipped {skipped}."
            )

    def _load_mel(self, wave_path: str):
        if self.dump_root is not None and self.dump_root.endswith(".parquet"):
            return _mel_from_store(self.dump_root, wave_path)
        if self.dump_root is None:
            raise RuntimeError("dump_root is None: mel-only mode requires pre-computed mel files")
            
        mel_key = osp.splitext(wave_path)[0].replace("/", "_").replace("\\", "_")
        if len(mel_key.encode('utf-8', 'ignore')) > 240:
            import hashlib
            mel_key = hashlib.md5(mel_key.encode('utf-8', 'ignore')).hexdigest()

        mel_path = None
        for split_dir in ["val", "train"]:
            candidate_path = osp.join(self.dump_root, split_dir, "mels", mel_key + ".pt")
            if os.path.exists(candidate_path):
                mel_path = candidate_path
                break

        if mel_path is None:
            expected_split = "val" if self.validation else "train"
            expected_path = osp.join(self.dump_root, expected_split, "mels", mel_key + ".pt")
            raise FileNotFoundError(f"Pre-computed mel not found: {expected_path}")

        try:
            raw = torch.load(mel_path, weights_only=True)
            if isinstance(raw, dict):
                raw = raw['mel']
            if raw.dim() == 3 and raw.shape[0] == 1:
                raw = raw.squeeze(0)
            mel = raw.float()
            if mel.dim() == 1:
                mel = mel.unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-computed mel: {mel_path} | {e}") from e

        length_feature = mel.size(1)
        mel = mel[:, : (length_feature - length_feature % 2)]
        return mel

    def _build_sample(self, idx: int, enforce_ratio_filter: bool = False):
        wave_path, text, language_id, speaker_id, quality_id, text_raw = self.data_list[idx]
        name = wave_path if not self.use_basename else osp.basename(wave_path)

        mel = self._load_mel(wave_path)
        if mel is None:
            raise RuntimeError(f"Mel load returned None for: {wave_path}")

        ids = self.text_cleaner(text)

        BOS_EOS = dicts["$"]  # should be 1
        SIL = dicts[" "]      # silence id

        # Prevent double silences if text already starts/ends with ' '
        while len(ids) > 0 and ids[0] == SIL:
            ids = ids[1:]
        while len(ids) > 0 and ids[-1] == SIL:
            ids = ids[:-1]

        # Add boundary markers + explicit boundary silences + trailing blank token (idx 0)
        ids = [BOS_EOS, SIL] + ids + [SIL, BOS_EOS]

        text_tensor = torch.LongTensor(ids)

        if enforce_ratio_filter and mel.shape[1] < (self.min_mel_text_ratio * text_tensor.shape[0]):
            return None

        raw_phonemes = text
        return mel, text_tensor, raw_phonemes, name

    def __len__(self):
        if self.cached_samples is not None:
            return len(self.cached_samples)
        return len(self.data_list)

    def __getitem__(self, idx: int):
        if self.cached_samples is not None:
            return self.cached_samples[idx]

        sample = self._build_sample(idx, enforce_ratio_filter=False)
        if sample is None:
            raise RuntimeError(f"Failed to build sample at index {idx}")
        return sample



class Collater:
    def __init__(self, expect_wave: bool = False, list_file_path: Optional[str] = None):
        self.expect_wave = expect_wave
        self.list_file_path = list_file_path

    def __call__(self, batch):
        # 1. Filter the batch first! 
        # We create a new list containing ONLY valid samples.
        filtered_batch = []
        for x in batch:
            mel, txt, ph, name = x
            mel_len = mel.shape[1]
            txt_len = txt.shape[0]

            # Check alignment condition
            # User requested 1.25x threshold
            if mel_len < (1.5 * txt_len):
                print(
                    f"SKIPPING IMPOSSIBLE ALIGNMENT: {name} | "
                    f"Mel Frames: {mel_len}, Text Tokens: {txt_len} | "
                    f"Need at least {int(1.5 * txt_len)} frames. Deleting from list..."
                )
                
                # Attempt to delete from the list file if path is provided
                if self.list_file_path and os.path.exists(self.list_file_path):
                    try:
                        import fcntl
                        with open(self.list_file_path, 'r+') as f:
                            fcntl.flock(f, fcntl.LOCK_EX)
                            lines = f.readlines()
                            f.seek(0)
                            f.truncate()
                            for line in lines:
                                parts = line.strip().split('|')
                                if len(parts) > 0 and parts[0] == name:
                                    continue # Skip this line
                                f.write(line)
                            fcntl.flock(f, fcntl.LOCK_UN)
                    except Exception as e:
                        print(f"Failed to delete {name} from {self.list_file_path}: {e}")

                continue  # This actually skips adding it to filtered_batch
            
            filtered_batch.append(x)

        # Update the batch to use only the filtered data
        batch = filtered_batch

        # 2. Handle edge case: If everything was filtered out, return None
        # (Your training loop must handle receiving None!)
        if len(batch) == 0:
            return None

        # 3. Now it is safe to calculate dimensions based on the valid batch
        B = len(batch)
        
        # Sort by mel length (descending) for efficiency
        lengths = [b[0].shape[1] for b in batch]
        order = np.argsort(lengths)[::-1]
        batch = [batch[i] for i in order]

        mel0 = batch[0][0]
        nmels = mel0.size(0)
        max_mel_len = max(b[0].shape[1] for b in batch)
        max_txt_len = max(b[1].shape[0] for b in batch)

        # Allocate tensors with the new Batch Size (B)
        mels = torch.zeros((B, nmels, max_mel_len), device=mel0.device).float()
        texts = torch.full((B, max_txt_len), fill_value=0).long()
        mel_lengths = torch.zeros((B,), dtype=torch.long)
        text_lengths = torch.zeros((B,), dtype=torch.long)

        raw_phonemes: List[str] = []
        names: List[str] = []
        waves = [] if self.expect_wave else None

        # 4. Fill tensors (No checks needed here, we already filtered)
        for i, (mel, txt, ph, name) in enumerate(batch):
            Tm, Tu = mel.shape[1], txt.shape[0]
            
            mels[i, :, :Tm] = mel
            texts[i, :Tu] = txt
            mel_lengths[i] = Tm
            text_lengths[i] = Tu
            
            raw_phonemes.append(ph)
            names.append(name)
            
            # (Optional) Wave handling if you added it back to __getitem__
            # if self.expect_wave:
            #     waves.append(wave)

        return SimpleNamespace(
            mels=mels,
            mel_length=mel_lengths,
            text=texts,
            text_length=text_lengths,
            raw_phonemes=raw_phonemes,
            names=names,
            waves=waves,
        )


def build_dataloader(
    path_list_lines: List[str],
    root_path: str,
    dump_root: Optional[str] = None, 
    validation: bool = False,
    batch_size: int = 8,
    num_workers: int = 2,
    device: str = "cpu",
    use_basename: bool = False,
    return_wave: bool = False,
    list_path: Optional[str] = None, # <--- Added argument
    cache_all_mels_in_ram: bool = False,
):
    effective_num_workers = num_workers
    if cache_all_mels_in_ram and num_workers != 0:
        logger.info("cache_all_mels_in_ram=True, forcing num_workers=0 for shared in-process RAM cache")
        effective_num_workers = 0

    dataset = FilePathDataset(
        path_list_lines,
        root_path=root_path,
        dump_root=dump_root, # Pass it down
        sr=SAMPLE_RATE,
        validation=validation,
        use_basename=use_basename,
        return_wave=return_wave,
        cache_all_mels_in_ram=cache_all_mels_in_ram,
    )
    collate_fn = Collater(expect_wave=return_wave, list_file_path=list_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not validation,
        num_workers=effective_num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu") and not _MELS_ON_GPU,  # GPU mels: batches are already on the GPU
    )
