#!/usr/bin/env python
"""One command: folder of name.wav|flac|mp3|opus + name.txt  ->  prepare -> align with a trained model -> JSON + TextGrid.

usage: align_folder.py --input DIR --model model/latest.pt --lang en-us --out OUT [--work WORK]
Clips must be 1-20 s (others are skipped). The model must be trained for the language(s) of the transcripts.
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGNER = os.path.join(HERE, "..", "aligner")


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", required=True, help="espeak-ng voice, e.g. en-us, pl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="scratch dir (default OUT/work)")
    ap.add_argument("--min-dur", type=int, default=3)
    ap.add_argument("--min-pause", type=int, default=5)
    ap.add_argument("--hmm-prior-scale", type=float, default=1.0)
    ap.add_argument("--format", default="folder", choices=["folder", "libritts", "mls"])
    a = ap.parse_args()
    py = sys.executable
    work = os.path.abspath(a.work or os.path.join(a.out, "work"))
    os.makedirs(a.out, exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)  # always start from a clean scratch dir
    run([py, os.path.join(HERE, "prepare_data.py"), "--input", a.input, "--format", a.format, "--lang", a.lang,
         "--out", work, "--val-frac", "1.0"])  # every clip goes to the "val" split, which is the one that is aligned
    run([py, "stylish_aligner_align.py", "--data-root", work, "--dump-root", f"{work}/audio.parquet",
         "--train-list", f"{work}/train_list.txt", "--val-list", f"{work}/val_list.txt",
         "--model-path", os.path.abspath(a.model), "--output-dir", f"{work}/align", "--decode", "hmm",
         "--min-dur", str(a.min_dur), "--hmm-prior-scale", str(a.hmm_prior_scale), "--min-pause", str(a.min_pause)],
        cwd=ALIGNER)
    run([py, os.path.join(HERE, "export_alignments.py"), "--align-dir", f"{work}/align", "--split", "val",
         "--list", f"{work}/val_list.txt", "--audio", f"{work}/audio.parquet", "--out", a.out, "--textgrid"])


if __name__ == "__main__":
    main()
