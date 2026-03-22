"""
build_metadata.py
=================
Walks the spectrograms directory and produces metadata.csv with one row
per spectrogram image.

Output columns:
    file_path       – relative path from spec_root, e.g.
                      001.Black_footed_Albatross/Black_footed_Albatross_116123_seg0.png
    class_id        – integer parsed from folder prefix  (1, 2, …)
    species         – human-readable name, e.g. "Black footed Albatross"
    image_id        – base recording id, e.g. "Black_footed_Albatross_116123"
    segment_number  – integer segment index (0 for unsegmented files)

Usage:
    python build_metadata.py --spec_root ./spectrograms
    python build_metadata.py --spec_root ./spectrograms --out metadata.csv
"""

import argparse
import csv
import re
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

# matches optional trailing _seg<N>  before the extension
_SEG_RE = re.compile(r"^(.+?)(?:_seg(\d+))?$")


def folder_to_species(folder_name: str) -> tuple[int, str]:
    """
    '001.Black_footed_Albatross'  →  (1, 'Black footed Albatross')
    """
    parts = folder_name.split(".", 1)          # ['001', 'Black_footed_Albatross']
    class_id = int(parts[0])
    species  = parts[1].replace("_", " ") if len(parts) > 1 else folder_name
    return class_id, species


def parse_stem(stem: str) -> tuple[str, int]:
    """
    'Black_footed_Albatross_116123_seg2'  →  ('Black_footed_Albatross_116123', 2)
    'Black_footed_Albatross_116123'        →  ('Black_footed_Albatross_116123', 0)
    """
    m = _SEG_RE.match(stem)
    image_id       = m.group(1)
    segment_number = int(m.group(2)) if m.group(2) is not None else 0
    return image_id, segment_number


# ── main ─────────────────────────────────────────────────────────────────────

def build_metadata(spec_root: Path, out_path: Path) -> None:
    rows = []

    for species_dir in sorted(spec_root.iterdir()):
        if not species_dir.is_dir():
            continue
        try:
            class_id, species = folder_to_species(species_dir.name)
        except (ValueError, IndexError):
            print(f"[SKIP] Cannot parse folder name: {species_dir.name}")
            continue

        for img_file in sorted(species_dir.glob("*.png")):
            image_id, segment_number = parse_stem(img_file.stem)
            file_path = f"{species_dir.name}/{img_file.name}"
            rows.append({
                "file_path":      file_path,
                "class_id":       class_id,
                "species":        species,
                "image_id":       image_id,
                "segment_number": segment_number,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_path", "class_id", "species",
                           "image_id", "segment_number"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓  Wrote {len(rows):,} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_root", type=Path, default=Path("spectrograms"),
                        help="Root folder of spectrogram images.")
    parser.add_argument("--out", type=Path, default=Path("metadata.csv"),
                        help="Output CSV path.")
    args = parser.parse_args()

    if not args.spec_root.exists():
        print(f"[ERROR] spec_root not found: {args.spec_root}")
        raise SystemExit(1)

    build_metadata(args.spec_root, args.out)


if __name__ == "__main__":
    main()