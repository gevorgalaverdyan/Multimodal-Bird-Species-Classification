"""
Audio Preprocessing Pipeline
Converts MP3 recordings into 224x224 colour spectrograms for ResNet-50 input.

Preprocessing follows the methodology of:
  Naranchimeg et al. (2018) "Cross-domain Deep Feature Combination for Bird
  Species Classification with Audio-visual Data" (arXiv:1811.10199)

Exact pipeline per recording:
  1. Load MP3, resample to 22050 Hz mono
  2. Trim leading/trailing silence
  3. Fix to exactly 10 seconds (centre-crop or zero-pad)
  4. Noise reduction: zero out frames where max amplitude < 1/4 of recording's
     peak amplitude  (paper Section 4: "removed a frame which contains only
     amplitude less than 1/4 of the maximum amplitude")
  5. STFT with Hanning window size=512, 50% overlap (hop=256)
     frequency range 0–10 kHz  (paper: "sounds of birds are usually contained
     in a small portion of the frequency range, mostly around 2–8 kHz, so we
     only extract features from the range of (0, 10) kHz")
  6. Convert to Mel spectrogram → log (dB) scale → normalize to [0, 1]
  7. Apply colour map → resize to 224×224 RGB PNG

Output mirrors input structure:
  spectrograms/
  └── 001.Black_footed_Albatross/
      └── Black_footed_Albatross_694038.png
"""

import os
import warnings
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration (matching Naranchimeg et al. 2018) ─────────────────────────

DATA_DIR    = "../audio-scraper/data"
OUTPUT_DIR  = "./spectrograms"

TARGET_SR   = 22050     # resample rate (Hz)
DURATION    = 10.0      # fixed clip length in seconds (paper: 10s frames)
N_FFT       = 512       # Hanning window size (paper: "windowed with Hanning window size 512")
HOP_LENGTH  = 256       # 50% overlap  (paper: "50% overlap")
N_MELS      = 128       # mel filterbank bins
F_MIN       = 0         # lower frequency bound (Hz)
F_MAX       = 10000     # upper frequency bound (Hz) — paper: "(0, 10) kHz"
NOISE_RATIO = 1 / 4     # paper: "amplitude less than 1/4 of the maximum amplitude"
IMG_SIZE    = 224       # ResNet-50 input (paper used 227 for CaffeNet)
COLORMAP    = "viridis" # discriminant colour map

# ── Pipeline steps ────────────────────────────────────────────────────────────

def load_and_resample(path: str) -> np.ndarray:
    """Load MP3 as mono float32, resampled to TARGET_SR."""
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return y


def trim_silence(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Remove leading and trailing silence."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def fix_duration(y: np.ndarray) -> np.ndarray:
    """
    Clip or zero-pad to exactly DURATION seconds.
    Clips from the centre — maximises chance of capturing the vocalization.
    """
    target_len = int(TARGET_SR * DURATION)
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        return y[start: start + target_len]
    pad = target_len - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2), mode="constant")


def remove_noise_frames(y: np.ndarray) -> np.ndarray:
    """
    Zero out frames whose maximum amplitude is below NOISE_RATIO * peak amplitude.

    Matches paper Section 4:
      "we obtained the maximum amplitude of the audio and removed a frame
       which contains only amplitude less than 1/4 of the maximum amplitude"
    """
    peak_amplitude = np.max(np.abs(y))
    threshold = NOISE_RATIO * peak_amplitude

    y_clean = y.copy()
    n_samples = len(y)

    for start in range(0, n_samples, HOP_LENGTH):
        end = min(start + N_FFT, n_samples)
        frame = y[start:end]
        if np.max(np.abs(frame)) < threshold:
            y_clean[start:end] = 0.0

    return y_clean


def compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Compute log-scale Mel spectrogram, normalized to [0, 1].

    STFT parameters match paper:
      - Hanning window, size N_FFT=512
      - 50% overlap → hop_length=256
      - frequency range 0–10 kHz
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann",          # Hanning window (paper)
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0,
    )
    S_db   = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm


def spectrogram_to_image(S_norm: np.ndarray) -> Image.Image:
    """Render normalised spectrogram as a 224×224 colour PNG."""
    dpi      = 100
    fig_size = IMG_SIZE / dpi

    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.imshow(
        S_norm,
        aspect="auto",
        origin="lower",
        cmap=COLORMAP,
        interpolation="bilinear",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    buf       = fig.canvas.buffer_rgba() # type: ignore
    img_array = np.asarray(buf)
    plt.close(fig)

    img = Image.fromarray(img_array).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS) # type: ignore
    return img


def process_file(mp3_path: str, out_path: str) -> bool:
    """Full pipeline for one MP3 → PNG. Returns True on success."""
    try:
        y = load_and_resample(mp3_path)
        y = trim_silence(y)
        y = fix_duration(y)
        y = remove_noise_frames(y)
        S = compute_mel_spectrogram(y)
        img = spectrogram_to_image(S)
        img.save(out_path)
        return True
    except Exception as e:
        print(f"\n  ERROR {mp3_path}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_path = Path(DATA_DIR)
    out_path  = Path(OUTPUT_DIR)

    species_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not species_dirs:
        raise RuntimeError(f"No species folders found in {DATA_DIR}")

    # Collect jobs: only MP3s whose PNG does not yet exist (resumable)
    jobs = []
    for species_dir in species_dirs:
        out_species_dir = out_path / species_dir.name
        out_species_dir.mkdir(parents=True, exist_ok=True)
        for mp3 in sorted(species_dir.glob("*.mp3")):
            png = out_species_dir / (mp3.stem + ".png")
            if not png.exists():
                jobs.append((str(mp3), str(png)))

    if not jobs:
        print("All spectrograms already generated — nothing to do.")
        return

    print(f"Found {len(jobs)} MP3s to convert across {len(species_dirs)} species")
    print(f"Settings: {DURATION}s clips | Hanning window {N_FFT} | hop {HOP_LENGTH} "
          f"(50% overlap) | 0–{F_MAX//1000}kHz | noise threshold 1/{int(1/NOISE_RATIO)} peak amp\n")

    success, failed = 0, []

    for mp3_path, png_path in tqdm(jobs, unit="file"):
        if process_file(mp3_path, png_path):
            success += 1
        else:
            failed.append(mp3_path)

    print(f"\n✓ {success}/{len(jobs)} spectrograms saved to {OUTPUT_DIR}/")
    if failed:
        print(f"✗ {len(failed)} failed:")
        for f in failed:
            print(f"    {f}")


if __name__ == "__main__":
    main()