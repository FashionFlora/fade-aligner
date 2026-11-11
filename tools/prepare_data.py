#!/usr/bin/env python
"""Build the data the Fade Aligner reads: OUT/train_list.txt, OUT/val_list.txt and OUT/audio.parquet.

  list line : relative/key.flac|IPA phonemes|lang_id|speaker_id|0|original text
  audio.parquet : columns `key` (the list key without ".flac") and `audio` (44.1 kHz mono FLAC bytes); the aligner
                  computes its mels from this file.

Input formats (--format):
  folder    DIR with name.wav|flac|mp3|opus + name.txt pairs (any sample rate; sub-folders = speakers)
  libritts  LibriTTS / LibriTTS-R:  DIR/.../speaker/chapter/*.wav + *.normalized.txt
  mls       Multilingual LibriSpeech: DIR/(train/)audio/speaker/book/*.opus|flac + transcripts.txt

All audio is resampled to 44.1 kHz (soxr). Corpora recorded at lower rates (24 kHz, 16 kHz) are upsampled: the mels then
carry no energy above the original Nyquist frequency, which the aligner tolerates but does not make better.

Several runs can be combined into one multi-language set: run once per source with the same --out and --append.
`merge` (python prepare_data.py merge OUT A B ...) concatenates finished output directories instead.

--lang is an espeak-ng voice (en-us, pl, de, fr-fr, ...). The lang_id list field is bookkeeping only: the aligner reads
and ignores it (the phoneme symbol table is shared by all languages). en-us=0, fr-fr=1, de=2, ru=3, pl=4, es=5, it=10,
pt=11; any other voice gets the next free id (>= 12), or set --lang-id.
"""
import argparse, glob, io, json, os, random, re, sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SR = 44100
KNOWN_IDS = {"en-us": 0, "fr-fr": 1, "de": 2, "ru": 3, "pl": 4, "es": 5, "it": 10, "pt": 11}
AUDIO_EXT = (".wav", ".flac", ".mp3", ".opus")


# ----------------------------------------------------------------------------- input formats
def read_text(p):
    with open(p, encoding="utf-8") as f:
        return " ".join(f.read().split())


def list_folder(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            b, e = os.path.splitext(fn)
            t = os.path.join(dp, b + ".txt")
            if e.lower() in AUDIO_EXT and os.path.exists(t):
                rel = os.path.relpath(dp, root)
                out.append((os.path.join(dp, fn), read_text(t), "spk0" if rel == "." else rel.split(os.sep)[0], b if rel == "." else f"{rel}/{b}"))
    return out


def list_libritts(root):
    out = []
    for t in glob.glob(os.path.join(root, "**", "*.normalized.txt"), recursive=True):
        w = t[: -len(".normalized.txt")] + ".wav"
        if os.path.exists(w):
            parts = w.split(os.sep)
            out.append((w, read_text(t), parts[-3], f"{parts[-3]}/{parts[-2]}/{parts[-1][:-4]}"))
    return out


def list_mls(root):
    out = []
    base = root
    for cand in (os.path.join(root, "train"), root):
        if os.path.exists(os.path.join(cand, "transcripts.txt")):
            base = cand
            break
    else:
        hits = glob.glob(os.path.join(root, "**", "transcripts.txt"), recursive=True)
        if not hits:
            sys.exit("transcripts.txt not found under " + root)
        base = os.path.dirname(sorted(hits)[0])
    with open(os.path.join(base, "transcripts.txt"), encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            uid, txt = line.rstrip("\n").split("\t", 1)
            spk, book, _ = uid.split("_", 2)
            for ext in (".opus", ".flac"):
                p = os.path.join(base, "audio", spk, book, uid + ext)
                if os.path.exists(p):
                    out.append((p, " ".join(txt.split()), spk, f"{spk}/{book}/{uid}"))
                    break
    return out


# ----------------------------------------------------------------------------- audio
def load_clip(args):
    """path -> (flac bytes at 44.1 kHz, duration s) or None (unreadable / outside the duration range)"""
    path, lo, hi = args
    try:
        info = sf.info(path)
        d0 = info.frames / info.samplerate
        if not lo <= d0 <= hi:
            return None
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        try:  # e.g. mp3/opus without libsndfile support: fall back to librosa (audioread / ffmpeg)
            import librosa
            x, sr = librosa.load(path, sr=None, mono=False)
            x = x.T if x.ndim > 1 else x[:, None]
        except Exception:
            return None
    x = x.mean(axis=1)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR, quality="HQ")
    x = np.clip(x, -1.0, 1.0)
    d = len(x) / SR
    if not lo <= d <= hi:
        return None
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue(), d


# ----------------------------------------------------------------------------- merge
def merge(out, srcs):
    os.makedirs(out, exist_ok=True)
    w = None
    for s in srcs:
        pf = pq.ParquetFile(f"{s}/audio.parquet")
        for rb in pf.iter_batches(batch_size=512):
            t = pa.Table.from_batches([rb])
            w = w or pq.ParquetWriter(f"{out}/audio.parquet", t.schema, compression="none")
            w.write_table(t)
    w.close()
    for sp in ("train", "val"):
        with open(f"{out}/{sp}_list.txt", "w", encoding="utf-8") as f:
            for s in srcs:
                f.write(open(f"{s}/{sp}_list.txt", encoding="utf-8").read())
    print("merged", srcs, "->", out, "(speaker ids of the sources may overlap; they are not used by the aligner)")


# ----------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3:])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="corpus directory")
    ap.add_argument("--format", choices=["folder", "libritts", "mls"], default="folder")
    ap.add_argument("--lang", required=True, help="espeak-ng voice, e.g. en-us, pl, de")
    ap.add_argument("--lang-id", type=int, default=None, help="override the list lang_id (informational)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true", help="add to an existing --out (other language / source)")
    ap.add_argument("--max-hours", type=float, default=float("inf"))
    ap.add_argument("--max-clips", type=int, default=0, help="0 = no limit (handy for smoke tests)")
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prefix", default=None, help="key prefix, unique per source (default: the language)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    items = {"folder": list_folder, "libritts": list_libritts, "mls": list_mls}[a.format](a.input)
    items = [i for i in items if i[1].strip() and "|" not in i[1] + i[3]]
    rng.shuffle(items)  # random order = speakers and chapters spread over the selection
    print(f"{len(items)} candidate clips", flush=True)
    prefix = a.prefix or a.lang

    os.makedirs(a.out, exist_ok=True)
    meta_path = f"{a.out}/meta.json"
    meta = json.load(open(meta_path)) if a.append and os.path.exists(meta_path) else {"langs": {}, "speakers": {}, "sources": []}
    if a.lang_id is not None:
        lang_id = a.lang_id
    elif a.lang in meta["langs"]:
        lang_id = meta["langs"][a.lang]
    elif a.lang in KNOWN_IDS:
        lang_id = KNOWN_IDS[a.lang]
    else:
        lang_id = max([11] + list(meta["langs"].values())) + 1
    meta["langs"][a.lang] = lang_id

    store_tmp = f"{a.out}/audio.parquet.tmp"
    schema = pa.schema([("key", pa.string()), ("audio", pa.binary())])
    writer = pq.ParquetWriter(store_tmp, schema, compression="none")
    old_lists = {"train": [], "val": []}
    if a.append and os.path.exists(f"{a.out}/audio.parquet"):
        for rb in pq.ParquetFile(f"{a.out}/audio.parquet").iter_batches(batch_size=512):
            writer.write_table(pa.Table.from_batches([rb], schema=schema))
        for sp in old_lists:
            old_lists[sp] = [l for l in open(f"{a.out}/{sp}_list.txt", encoding="utf-8").read().split("\n") if l]
    elif os.path.exists(f"{a.out}/audio.parquet"):
        sys.exit(f"{a.out}/audio.parquet exists; use --append or another --out")

    picked, sec = [], 0.0
    keys_b, blobs_b = [], []
    seen = {l.split("|")[0] for sp in old_lists.values() for l in sp}
    with Pool(a.workers) as pool:
        for it, r in zip(items, pool.imap(load_clip, [(i[0], a.min_s, a.max_s) for i in items], chunksize=8)):
            if r is None:
                continue
            key = f"{prefix}/{re.sub(r'[^A-Za-z0-9_./-]', '_', it[3])}"
            if key + ".flac" in seen:
                continue
            seen.add(key + ".flac")
            keys_b.append(key); blobs_b.append(r[0])
            picked.append((key, it[1], it[2]))
            sec += r[1]
            if len(keys_b) >= 512:
                writer.write_table(pa.table({"key": keys_b, "audio": pa.array(blobs_b, pa.binary())}, schema=schema))
                keys_b, blobs_b = [], []
            if len(picked) % 2000 == 0:
                print(f"  {len(picked)} clips, {sec / 3600:.1f} h", flush=True)
            if sec >= a.max_hours * 3600 or (a.max_clips and len(picked) >= a.max_clips):
                pool.terminate()
                break
    if keys_b:
        writer.write_table(pa.table({"key": keys_b, "audio": pa.array(blobs_b, pa.binary())}, schema=schema))
    writer.close()
    print(f"{len(picked)} clips, {sec / 3600:.2f} h of {a.lang}", flush=True)

    # --- phonemes: espeak-ng output as is (settings must match what the model was trained with)
    from phonemizer.backend import EspeakBackend
    be = EspeakBackend(language=a.lang, preserve_punctuation=True, with_stress=True,
                       language_switch="remove-flags", words_mismatch="ignore")
    ph = be.phonemize([p[1] for p in picked], strip=True, njobs=4)

    new = {"train": [], "val": []}
    for (key, text, spk), p in zip(picked, ph):
        if not p.strip():
            continue
        sid = meta["speakers"].setdefault(f"{a.lang}:{spk}", len(meta["speakers"]))
        sp = "val" if rng.random() < a.val_frac else "train"
        new[sp].append("|".join([key + ".flac", p, str(lang_id), str(sid), "0", text]))
    for sp in new:
        with open(f"{a.out}/{sp}_list.txt", "w", encoding="utf-8") as f:
            L = old_lists[sp] + new[sp]
            f.write("\n".join(L) + "\n" if L else "")  # a truly empty file when the split has no clips
    os.replace(store_tmp, f"{a.out}/audio.parquet")
    meta["sources"].append({"lang": a.lang, "format": a.format, "clips": len(picked), "hours": round(sec / 3600, 2), "seed": a.seed})
    json.dump(meta, open(meta_path, "w"), indent=1)
    print("DONE train", len(old_lists["train"]) + len(new["train"]), "val", len(old_lists["val"]) + len(new["val"]), flush=True)


if __name__ == "__main__":
    main()
