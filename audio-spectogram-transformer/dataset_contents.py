"""
build_dataset_contents.py
=========================
Produces dataset_contents.csv — one row per species — summarising how many
spectrogram images exist and which source audio files they came from.

Output columns:
    class_id  – integer (1–200)
    species   – human-readable name, e.g. "Black footed Albatross"
    folder    – folder name, e.g. "001.Black_footed_Albatross"
    count     – number of spectrogram PNG images found
    status    – "OK" if count > 0, else "MISSING"
    files     – comma-separated list of source MP3 filenames (derived from
                image stems), or empty string if MISSING

The script discovers all species from BOTH the spectrogram folder and the
audio folder (so species that produced zero spectrograms still appear as
MISSING rows).  Pass only one of the two if you don't have the other.

Usage:
    python build_dataset_contents.py \
        --spec_root  ./spectrograms \
        --audio_root ../audio-scraper/data \
        --out dataset_contents.csv
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

_SEG_RE = re.compile(r"^(.+?)(?:_seg\d+)?$")


def folder_to_meta(folder_name: str) -> tuple[int, str]:
    """'002.Laysan_Albatross' → (2, 'Laysan Albatross')"""
    parts = folder_name.split(".", 1)
    class_id = int(parts[0])
    species  = parts[1].replace("_", " ") if len(parts) > 1 else folder_name
    return class_id, species


def image_stem_to_mp3(stem: str) -> str:
    """
    Strip trailing _seg<N> and append .mp3.
    'Black_footed_Albatross_116123_seg2' → 'Black_footed_Albatross_116123.mp3'
    'Black_footed_Albatross_116123'       → 'Black_footed_Albatross_116123.mp3'
    """
    m = _SEG_RE.match(stem)
    return (m.group(1) if m else stem) + ".mp3"


# ── collectors ────────────────────────────────────────────────────────────────

def collect_from_spectrograms(spec_root: Path) -> dict[str, dict]:
    """
    Returns {folder_name: {class_id, species, count, files_set}}
    """
    data = {}
    for species_dir in sorted(spec_root.iterdir()):
        if not species_dir.is_dir():
            continue
        try:
            class_id, species = folder_to_meta(species_dir.name)
        except (ValueError, IndexError):
            continue

        png_files = sorted(species_dir.glob("*.png"))
        # derive unique source mp3 names from image stems
        source_mp3s = sorted({image_stem_to_mp3(p.stem) for p in png_files})

        data[species_dir.name] = {
            "class_id": class_id,
            "species":  species,
            "folder":   species_dir.name,
            "count":    len(png_files),
            "files":    source_mp3s,      # list of mp3 names
        }
    return data


def collect_from_audio(audio_root: Path) -> dict[str, dict]:
    """
    Returns {folder_name: {class_id, species, files_set}}
    Useful for discovering species that have audio but zero spectrograms.
    """
    data = {}
    for species_dir in sorted(audio_root.iterdir()):
        if not species_dir.is_dir():
            continue
        try:
            class_id, species = folder_to_meta(species_dir.name)
        except (ValueError, IndexError):
            continue

        mp3s = sorted(p.name for p in species_dir.iterdir()
                      if p.suffix.lower() == ".mp3")
        data[species_dir.name] = {
            "class_id": class_id,
            "species":  species,
            "folder":   species_dir.name,
            "files":    mp3s,
        }
    return data


# ── main ─────────────────────────────────────────────────────────────────────

def build_dataset_contents(spec_root: Path | None,
                            audio_root: Path | None,
                            out_path: Path) -> None:

    spec_data  = collect_from_spectrograms(spec_root)  if spec_root  else {}
    audio_data = collect_from_audio(audio_root)        if audio_root else {}

    # union of all known folder names
    all_folders = sorted(
        set(spec_data.keys()) | set(audio_data.keys()),
        key=lambda f: int(f.split(".")[0])          # sort by class_id
    )

    rows = []
    for folder in all_folders:
        s = spec_data.get(folder, {})
        a = audio_data.get(folder, {})

        # prefer spectrogram-derived metadata; fall back to audio-derived
        class_id = s.get("class_id") or a.get("class_id", 0)
        species  = s.get("species")  or a.get("species",  folder)
        count    = s.get("count", 0)
        status   = "OK" if count >= 30 else "LOW" if count > 0 else "MISSING"

        # files: from spectrograms (derived mp3 names) if available,
        #        else from audio folder directly
        files_list = s.get("files") or a.get("files") or []
        files_str  = ", ".join(files_list)

        rows.append({
            "class_id": class_id,
            "species":  species,
            "folder":   folder,
            "count":    count,
            "status":   status,
            "files":    files_str,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_id", "species", "folder",
                           "count", "status", "files"])
        writer.writeheader()
        writer.writerows(rows)

    ok      = sum(1 for r in rows if r["status"] == "OK")
    low     = sum(1 for r in rows if r["status"] == "LOW")
    missing = sum(1 for r in rows if r["status"] == "MISSING")
    total   = sum(r["count"] for r in rows)
    print(f"✓  {len(rows)} species  |  {ok} OK  |  {low} LOW  |  {missing} MISSING  "
          f"|  {total:,} total spectrograms  →  {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_root",  type=Path,
                        default=Path("spectrograms"),
                        help="Root folder of spectrogram images.")
    parser.add_argument("--audio_root", type=Path,
                        default=Path("../audio-scraper/data"),
                        help="Root folder of source audio (for MISSING detection).")
    parser.add_argument("--out", type=Path,
                        default=Path("dataset_contents.csv"),
                        help="Output CSV path.")
    args = parser.parse_args()

    # validate paths
    spec_root  = args.spec_root  if args.spec_root.exists()  else None
    audio_root = args.audio_root if args.audio_root.exists() else None

    if spec_root is None and audio_root is None:
        print("[ERROR] Neither spec_root nor audio_root found.")
        raise SystemExit(1)

    if spec_root  is None: print(f"[WARN] spec_root not found:  {args.spec_root}")
    if audio_root is None: print(f"[WARN] audio_root not found: {args.audio_root}")

    build_dataset_contents(spec_root, audio_root, args.out)


if __name__ == "__main__":
    main()