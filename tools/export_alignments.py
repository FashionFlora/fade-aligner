#!/usr/bin/env python
"""Convert aligner output (alignments_*.safetensors) into per-clip JSON and optional Praat TextGrids.

JSON per clip: {"key","text","ipa","sr","duration_s","score","tokens":[[symbol,start_s,end_s],...]}
  symbol "_" = space / silence token, "<b>" = the two boundary markers (zero-width in the real audio).
Mel frames are 10 ms; the aligner pads the audio by 1764 samples (4 frames) on both sides before the mels, so 4 frames are
subtracted from every boundary and times are clipped to [0, duration].
TextGrid: tier "phones" (one interval per token, silence "_" kept) and tier "words" (tokens between "_" tokens, labelled
with the IPA word).

usage: export_alignments.py --align-dir DIR --list val_list.txt --audio audio.parquet --out OUT [--keys k1,k2 | --n N]
                            [--textgrid] [--save-audio]
(--align-dir holds alignments_<split>.safetensors and scores_<split>.txt; --split picks which, default val)
"""
import argparse, io, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aligner"))
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from safetensors import safe_open
from meldataset_preparation import TextCleaner

FRAME_S, PAD_FRAMES = 0.01, 4


def token_ids(ipa, tc):
    sil = tc.word_index_dictionary[" "]
    ids = list(tc(ipa))
    while ids and ids[0] == sil:
        ids = ids[1:]
    while ids and ids[-1] == sil:
        ids = ids[:-1]
    return [-1, sil] + ids + [sil, -1]  # same as the dataset: [BOS_EOS, SIL] + ids + [SIL, BOS_EOS]; -1 shown as <b>


def clip_tokens(dur, ipa, tc, audio_s):
    ids = token_ids(ipa, tc)
    if len(ids) != len(dur):
        return None
    sil = tc.word_index_dictionary[" "]
    inv = {v: k for k, v in tc.word_index_dictionary.items()}
    edges = np.concatenate([[0], np.cumsum(dur)])
    out = []
    for j, i in enumerate(ids):
        sym = "<b>" if i == -1 else "_" if i == sil else inv.get(int(i), "?")
        s = min(max((edges[j] - PAD_FRAMES) * FRAME_S, 0.0), audio_s)
        e = min(max((edges[j + 1] - PAD_FRAMES) * FRAME_S, 0.0), audio_s)
        out.append([sym, round(float(s), 4), round(float(e), 4)])
    return out


def words_of(tokens):
    """(label, start, end) per word: tokens between "_" tokens, markers excluded"""
    words, cur = [], []
    for sym, s, e in tokens + [["_", 0, 0]]:
        if sym == "<b>":
            continue
        if sym == "_":
            if cur:
                words.append(("".join(t[0] for t in cur), cur[0][1], cur[-1][2]))
            cur = []
        else:
            cur.append([sym, s, e])
    return words


def textgrid(tokens, dur_s):
    ph = [t for t in tokens if t[0] != "<b>" and t[2] > t[1]]
    ph = [("" if s == "_" else s, a, b) for s, a, b in ph]
    wd = []
    t = 0.0
    for lab, a, b in words_of([x for x in tokens if x[2] > x[1] or x[0] == "_"]):
        if a > t + 1e-6:
            wd.append(("", t, a))
        wd.append((lab, a, b)); t = b
    if dur_s > t + 1e-6:
        wd.append(("", t, dur_s))

    def tier(name, iv):
        s = f'    item [{{}}]:\n        class = "IntervalTier"\n        name = "{name}"\n        xmin = 0\n        xmax = {dur_s:.4f}\n        intervals: size = {len(iv)}\n'
        for k, (lab, a, b) in enumerate(iv, 1):
            s += f'        intervals [{k}]:\n            xmin = {a:.4f}\n            xmax = {b:.4f}\n            text = "{lab.replace(chr(34), chr(34) * 2)}"\n'
        return s
    return ('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\n'
            f'xmax = {dur_s:.4f}\ntiers? <exists>\nsize = 2\nitem []:\n'
            + tier("phones", ph).format(1) + tier("words", wd).format(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--list", required=True, help="list file of the split (text + IPA)")
    ap.add_argument("--audio", required=True, help="audio.parquet (for the clip duration / --save-audio)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default=None, help="comma-separated keys (as in the list, with or without .flac)")
    ap.add_argument("--n", type=int, default=0, help="export only the first N clips (0 = all)")
    ap.add_argument("--textgrid", action="store_true")
    ap.add_argument("--save-audio", action="store_true", help="also write each clip as FLAC next to its JSON")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lines = {}
    for l in open(a.list, encoding="utf-8"):
        p = l.rstrip("\n").split("|")
        if len(p) >= 6:
            lines[p[0]] = p
    scores = {}
    sp = os.path.join(a.align_dir, f"scores_{a.split}.txt")
    if os.path.exists(sp):
        for l in open(sp):
            v, k = l.split(None, 1)
            scores[k.strip()] = float(v)
    f = safe_open(os.path.join(a.align_dir, f"alignments_{a.split}.safetensors"), "pt")
    keys = [k if k.endswith(".flac") else k + ".flac" for k in a.keys.split(",")] if a.keys else [k for k in f.keys()]
    if a.n:
        keys = keys[: a.n]
    want = {k[:-5] for k in keys}
    audio = {}
    for rb in pq.ParquetFile(a.audio).iter_batches(batch_size=512, columns=["key", "audio"]):
        for k, b in zip(rb.column("key").to_pylist(), rb.column("audio").to_pylist()):
            if k in want:
                audio[k] = b
        if len(audio) == len(want):
            break

    tc = TextCleaner()
    for key in keys:
        if key not in lines or key[:-5] not in audio:
            print("skip (not in list / audio store):", key); continue
        x, sr = sf.read(io.BytesIO(audio[key[:-5]]), dtype="float32")
        dur_s = len(x) / sr
        d = f.get_tensor(key).numpy()
        toks = clip_tokens(d, lines[key][1], tc, dur_s)
        if toks is None:
            print("skip (token count mismatch):", key); continue
        name = key[:-5].replace("/", "__")
        head = {"key": key[:-5], "text": lines[key][5], "ipa": lines[key][1], "sr": sr, "duration_s": round(dur_s, 4),
                "score": scores.get(key)}
        body = ",\n".join("  " + json.dumps(t, ensure_ascii=False) for t in toks)
        with open(f"{a.out}/{name}.json", "w", encoding="utf-8") as fo:  # one token per line
            fo.write(json.dumps(head, ensure_ascii=False)[:-1] + ',\n "tokens": [\n' + body + "\n]}\n")
        if a.textgrid:
            open(f"{a.out}/{name}.TextGrid", "w", encoding="utf-8").write(textgrid(toks, dur_s))
        if a.save_audio:
            sf.write(f"{a.out}/{name}.flac", x, sr)
        frames_audio = (len(x) + 2 * 1764) // 441 + 1
        print(f"{name}: {len(toks)} tokens, frames {int(d.sum())} (audio ~{frames_audio}), {dur_s:.2f} s")


if __name__ == "__main__":
    main()
