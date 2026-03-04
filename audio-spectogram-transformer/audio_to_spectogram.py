"""
Spectrogram Generator for Bird Species Classification
======================================================
Reproduces the audio preprocessing pipeline from:
  "Cross-domain Deep Feature Combination for Bird Species Classification
   with Audio-visual Data" (Naranchimeg et al., 2018)

Methodology (exact match):
  - STFT over 10-second audio frames
  - Hanning window, size=512, 50% overlap (hop=256)
  - Frequency range: 0–10 kHz only
  - Silence removal: drop frames where max amplitude < 1/4 of global max
  - Output: 224×224 px color (viridis) spectrogram images

Dataset layout expected:
  <AUDIO_ROOT>/
    001.Black_footed_Albatross/
      Black_footed_Albatross_694038.mp3
      ...
    002.Laysan_Albatross/
      ...

Output mirrors the same layout under <SPEC_ROOT>.

Usage:
  python generate_spectrograms.py \
      --audio_root ./data \
      --spec_root  ./spectrograms \
      [--segment_sec 10] \
      [--workers 4]
"""

import argparse
import logging
import os
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────── constants (paper values) ──────────────────────
SR            = 22050          # resample target (standard; keeps ≥10 kHz)
SEGMENT_SEC   = 10             # seconds per spectrogram
N_FFT         = 512            # Hanning window size
HOP_LENGTH    = 256            # 50 % overlap
FMIN          = 0              # Hz  – lower bound of extracted range
FMAX          = 16_000         # Hz  – upper bound  (paper: 0–10 kHz)
SILENCE_RATIO = 0.25           # drop frames below 1/4 of max amplitude
IMG_SIZE      = (224, 224)     # final image dimensions (paper value)
AUDIO_EXTS    = {".mp3", ".wav", ".ogg", ".flac"}

# ───────────────────────────── logging setup ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Core processing
# ═══════════════════════════════════════════════════════════════════════════

def remove_silent_frames(y: np.ndarray, sr: int,
                         hop: int, silence_ratio: float) -> np.ndarray:
    """
    Frame the signal and drop frames whose peak amplitude is below
    `silence_ratio * global_max`.  Remaining frames are concatenated.

    This implements the paper's rule:
      "remove a frame which contains only amplitude less than 1/4 of
       the maximum amplitude"
    """
    if y.size == 0:
        return y

    global_max = np.max(np.abs(y))
    if global_max == 0:
        return y

    threshold = silence_ratio * global_max

    # Split into non-overlapping frames of `hop` samples for the energy check
    frames = librosa.util.frame(y, frame_length=hop, hop_length=hop)
    # frames shape: (hop, n_frames)
    frame_max = np.max(np.abs(frames), axis=0)          # peak per frame
    keep = frame_max >= threshold

    if not np.any(keep):
        log.debug("  All frames silent – keeping original signal.")
        return y

    kept_frames = frames[:, keep]                        # (hop, n_kept)
    return kept_frames.ravel(order="F")                  # back to 1-D


def audio_to_spectrogram_array(y: np.ndarray, sr: int) -> np.ndarray | None:
    """
    Compute a log-power spectrogram limited to [FMIN, FMAX] Hz.
    Returns a 2-D float array (freq_bins × time_frames), or None on failure.
    """
    if y.size < N_FFT:
        return None

    # ── STFT with Hanning window ──────────────────────────────────────────
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                             window="hann")) ** 2          # power spectrum

    # ── Frequency masking: keep only 0–10 kHz ────────────────────────────
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freq_mask = (freqs >= FMIN) & (freqs <= FMAX)
    S = S[freq_mask, :]

    if S.size == 0:
        return None

    # ── Convert to dB scale ───────────────────────────────────────────────
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def spectrogram_to_image(S_db: np.ndarray) -> Image.Image:
    """
    Render a spectrogram array as a 224×224 RGB colour image (viridis cmap),
    exactly matching the paper's stored format.
    """
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.imshow(S_db, aspect="auto", origin="lower",
              cmap="viridis", interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # render to an in-memory buffer (buffer_rgba works on matplotlib ≥3.8;
    # tostring_rgb was removed in 3.8)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())   # shape: (H, W, 4)  RGBA
    buf = buf[:, :, :3]                           # drop alpha → RGB
    plt.close(fig)

    img = Image.fromarray(buf, mode="RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img


def process_file(mp3_path: Path, out_path: Path,
                 segment_sec: int) -> tuple[str, bool, str]:
    """
    Full pipeline for a single audio file → one or more spectrogram images.

    Returns (relative_path, success, message).
    """
    rel = str(mp3_path)
    try:
        # ── Load & resample ───────────────────────────────────────────────
        y, sr = librosa.load(mp3_path, sr=SR, mono=True)

        # ── Remove silent frames ──────────────────────────────────────────
        y = remove_silent_frames(y, sr, hop=HOP_LENGTH,
                                 silence_ratio=SILENCE_RATIO)

        segment_samples = segment_sec * sr
        saved = 0

        if len(y) == 0:
            return rel, False, "Empty after silence removal"

        # ── Slice into non-overlapping 10-second segments ─────────────────
        # (A single recording may be shorter; we still use whatever is left)
        starts = range(0, max(1, len(y) - segment_samples + 1), segment_samples)

        for i, start in enumerate(starts):
            segment = y[start: start + segment_samples]

            # zero-pad the last (possibly short) segment to full length
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))

            S_db = audio_to_spectrogram_array(segment, sr)
            if S_db is None:
                continue

            img = spectrogram_to_image(S_db)

            # e.g. Black_footed_Albatross_694038_seg0.png
            stem = out_path.stem + (f"_seg{i}" if len(starts) > 1 else "")
            img_path = out_path.with_name(stem + ".png")
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)
            saved += 1

        if saved == 0:
            return rel, False, "No valid segments produced"

        return rel, True, f"{saved} segment(s) saved"

    except Exception as exc:
        return rel, False, str(exc)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset traversal
# ═══════════════════════════════════════════════════════════════════════════

def collect_jobs(audio_root: Path, spec_root: Path
                 ) -> list[tuple[Path, Path]]:
    """
    Walk audio_root, return list of (input_mp3, output_png_stem) pairs.
    Mirrors the folder structure under spec_root.
    """
    jobs = []
    for species_dir in sorted(audio_root.iterdir()):
        if not species_dir.is_dir():
            continue
        out_species_dir = spec_root / species_dir.name
        for audio_file in sorted(species_dir.iterdir()):
            if audio_file.suffix.lower() in AUDIO_EXTS:
                out_file = out_species_dir / audio_file.with_suffix(".png").name
                jobs.append((audio_file, out_file))
    return jobs


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert bird audio recordings to spectrograms "
                    "(Naranchimeg et al. 2018 pipeline).")
    parser.add_argument("--audio_root", type=Path, default=Path("../audio-scraper/data"),
                        help="Root folder containing species sub-directories.")
    parser.add_argument("--spec_root",  type=Path, default=Path("spectrograms"),
                        help="Output root (mirrors audio_root structure).")
    parser.add_argument("--segment_sec", type=int, default=SEGMENT_SEC,
                        help="Length of each spectrogram window in seconds (default 10).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default 4).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip audio files whose output already exists.")
    args = parser.parse_args()

    if not args.audio_root.exists():
        log.error("Audio root not found: %s", args.audio_root)
        raise SystemExit(1)

    args.spec_root.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(args.audio_root, args.spec_root)

    if args.skip_existing:
        jobs = [(src, dst) for src, dst in jobs if not dst.exists()]

    log.info("Found %d audio files to process.", len(jobs))
    log.info("Output root : %s", args.spec_root.resolve())
    log.info("Segment     : %d s  |  FFT: %d  |  Hop: %d  |  Freq: 0–%d Hz",
             args.segment_sec, N_FFT, HOP_LENGTH, FMAX)

    ok = fail = 0
    failures = []

    # ── Parallel processing ───────────────────────────────────────────────
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_file, src, dst, args.segment_sec): src
            for src, dst in jobs
        }
        for future in as_completed(futures):
            rel, success, msg = future.result()
            if success:
                ok += 1
                log.info("  ✓  %s  (%s)", Path(rel).name, msg)
            else:
                fail += 1
                failures.append((rel, msg))
                log.warning("  ✗  %s  — %s", Path(rel).name, msg)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Done.  ✓ %d succeeded   ✗ %d failed", ok, fail)

    if failures:
        log.info("Failed files:")
        for path, reason in failures:
            log.info("  • %s  →  %s", path, reason)

    # Write a simple report
    report_path = args.spec_root / "conversion_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Spectrogram generation report\n")
        f.write(f"Audio root  : {args.audio_root.resolve()}\n")
        f.write(f"Output root : {args.spec_root.resolve()}\n")
        f.write(f"Parameters  : segment={args.segment_sec}s, "
                f"n_fft={N_FFT}, hop={HOP_LENGTH}, "
                f"freq=0-{FMAX}Hz, silence_ratio={SILENCE_RATIO}\n")
        f.write(f"Total files : {ok + fail}\n")
        f.write(f"Succeeded   : {ok}\n")
        f.write(f"Failed      : {fail}\n\n")
        if failures:
            f.write("Failures:\n")
            for path, reason in failures:
                f.write(f"  {path}  →  {reason}\n")

    log.info("Report saved → %s", report_path)


if __name__ == "__main__":
    main()