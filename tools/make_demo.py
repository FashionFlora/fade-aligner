#!/usr/bin/env python
"""Build the demo data for the Fade Aligner project page.

Input  (--in): a folder with <key>.json + <key>.wav|.flac per clip. JSON:
  {"key", "text", "ipa", "sr", "duration_s", "score",
   "tokens": [[symbol, start_s, end_s], ...]}   ("_" = silence/space, "<b>" = boundary)
  Optional "lang". "phon" is accepted as an alias of "ipa".
Output (--out, usually docs): audio/<id>.mp3, spec/<id>.png, data/clips.json.

The spectrogram PNG spans exactly the audio duration (no axes, no margins), so
the page can overlay token boxes at start/duration percentages.

  python tools/make_demo.py --in export_dir --out docs --lang-map lang.json
--lang-map: a JSON file or an inline "prefix=lang,prefix=lang" string matched
against the clip key; otherwise the JSON "lang" field, otherwise a guess
(Polish letters -> pl, else en).
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PX_PER_S = 200          # spectrogram image width = duration * PX_PER_S
HEIGHT = 256
MP3_SR = 24000


def load_lang_map(arg):
    if not arg:
        return {}
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text())
    return dict(x.split("=", 1) for x in arg.split(",") if "=" in x)


def guess_lang(text):
    return "pl" if re.search(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", text) else "en"


def clean_tokens(tokens, dur):
    out = []
    for sym, a, b in tokens:
        if sym == "<b>":
            continue
        a = min(max(float(a), 0.0), dur)
        b = min(max(float(b), a), dur)
        out.append([sym, round(a, 4), round(b, 4)])
    return out


def spectrogram(y, sr, dur, path):
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    m = librosa.feature.melspectrogram(y=y16, sr=16000, n_fft=800, hop_length=80, n_mels=128,
                                       fmin=0, fmax=8000, power=2.0)
    db = librosa.power_to_db(m, ref=np.max, top_db=75)
    w = max(int(round(dur * PX_PER_S)), 16)
    fig = plt.figure(figsize=(w / 100, HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma", interpolation="bilinear",
              extent=(0, dur, 0, 1), vmin=-75, vmax=0)
    ax.set_xlim(0, dur); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(path, dpi=100, facecolor="black")
    plt.close(fig)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--lang-map", default="")
    a = ap.parse_args()
    src, out = Path(a.inp), Path(a.out)
    for d in ("audio", "spec", "data"):
        (out / d).mkdir(parents=True, exist_ok=True)
    lmap = load_lang_map(a.lang_map)
    clips, used = [], set()
    for jp in sorted(src.rglob("*.json")):
        meta = json.loads(jp.read_text())
        wav = next((jp.with_suffix(e) for e in (".wav", ".flac") if jp.with_suffix(e).exists()), None)
        if wav is None:
            print("no audio for", jp, file=sys.stderr); continue
        key = str(meta.get("key", jp.stem))
        cid = re.sub(r"[^A-Za-z0-9_-]+", "_", jp.stem).strip("_") or "clip"
        base, n = cid, 1
        while cid in used:
            n += 1; cid = f"{base}_{n}"
        used.add(cid)
        y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        dur = len(y) / sr
        text = meta.get("text", "")
        lang = meta.get("lang") or next((v for k, v in lmap.items() if key.startswith(k) or jp.stem.startswith(k)), None) or guess_lang(text)
        y24 = librosa.resample(y, orig_sr=sr, target_sr=MP3_SR) if sr != MP3_SR else y
        sf.write(str(out / "audio" / f"{cid}.mp3"), np.clip(y24, -1, 1), MP3_SR, format="MP3")
        spectrogram(y, sr, dur, out / "spec" / f"{cid}.png")
        clips.append({"id": cid, "lang": lang, "text": text, "ipa": meta.get("ipa", meta.get("phon", "")),
                      "audio": f"audio/{cid}.mp3", "spec": f"spec/{cid}.png", "duration": round(dur, 4),
                      "score": round(float(meta.get("score", 0.0)), 4),
                      "tokens": clean_tokens(meta["tokens"], dur)})
        print("ok", cid, lang, f"{dur:.2f}s", len(clips[-1]["tokens"]), "tokens")
    (out / "data" / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, separators=(",", ":")))
    print("wrote", len(clips), "clips to", out)


if __name__ == "__main__":
    main()
