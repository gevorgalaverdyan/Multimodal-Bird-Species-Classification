# Bird Species Audio Spectrograms (CUB-200 Aligned)

**10,553 spectrogram images · 199 species · 200 classes · sourced from Xeno-canto and eBird**

---

## Overview

This dataset provides audio spectrograms for **200 North American bird species**, aligned with the widely-used [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) image benchmark. It was assembled to support multimodal bird species classification research — specifically, studies that combine visual (image) and auditory (spectrogram) features under a shared CNN framework, following the methodology of:

Audio recordings were collected from [Xeno-canto](https://xeno-canto.org/) and [eBird](ebird.com), open source and community-driven repository of bird sounds. All recordings are freely available under Creative Commons licences.

---

## Dataset Statistics

OK: Count >= 30

| Metric | Value |
|---|---|
| Total spectrogram images | **10,553** |
| Species with data (OK) | **186 / 200** |
| Species with no recordings (MISSING) | **1 / 200** |
| Species with low ammount (LOW) | **13 / 200** |
| Spectrograms per species (min / mean / max) | **2 / 52 / 142** |
| Image size | **224 × 224 px** |
| Image format | **RGB PNG** |

---

## Spectrogram Generation Pipeline

Each MP3 recording was processed with the following steps, matching the paper's methodology (with adjustments for ResNet-50 input):

1. **Resample** to 22,050 Hz mono
2. **Silence removal** — signal is split into frames of 256 samples; any frame whose peak amplitude falls below ¼ of the global maximum is discarded, and the remaining frames are concatenated
3. **Segment** the cleaned signal into non-overlapping 10-second windows; the final window is zero-padded if shorter than 10 seconds
4. **STFT** with a Hanning window (size = 512, 50% overlap / hop = 256)
5. **Frequency masking** — retain only the 100 Hz – 16,000 Hz range
6. **Log-power (dB)** conversion via `librosa.power_to_db`
7. **Render** as a 224 × 224 px RGB image using the `viridis` colormap (ready for ResNet-50 / standard ImageNet-pretrained models)

Recordings longer than 10 seconds produce multiple segment images (named `_seg0`, `_seg1`, …). Shorter recordings produce a single image with no segment suffix.

The generation script (`audio_to_spectrogram.py`) is included in the dataset.

---

## File Structure

```
spectrograms/
├── 001.Black_footed_Albatross/
│   ├── Black_footed_Albatross_116349.png
│   ├── Black_footed_Albatross_116357_seg0.png
│   ├── Black_footed_Albatross_116357_seg1.png
│   └── ...
├── 002.Laysan_Albatross/
│   └── ...
│   ...
└── 200.Common_Yellowthroat/
    └── ...

metadata.csv
dataset_contents.csv
audio_to_spectrogram.py
```

Folder names follow the format `{class_id:03d}.{Species_Name}`, directly matching CUB-200-2011 conventions. The `class_id` is the integer index (1–200).

---

## Metadata Files

### `metadata.csv`
One row per spectrogram image.

| Column | Description |
|---|---|
| `file_path` | Relative path from dataset root, e.g. `001.Black_footed_Albatross/Black_footed_Albatross_116349_seg0.png` |
| `class_id` | Integer class label (1–200) |
| `species` | Human-readable species name, e.g. `Black footed Albatross` |
| `image_id` | Source recording identifier, e.g. `Black_footed_Albatross_116349` |
| `segment_number` | Segment index within the recording (0 for unsegmented files) |

## Missing Species

1 species from CUB-200-2011 has no audio data available on any provider and is an empty folder `003.Sooty_Albatross`. 

---

## Intended Use

This dataset is designed for:

- **Audio-only** bird species classification (treating spectrograms as images)
- **Multimodal learning** when paired with CUB-200-2011 images (same class IDs and species names)
- **Transfer learning** experiments with ImageNet-pretrained CNNs (ResNet-50, EfficientNet, ViT, etc.)

---

## Licence & Attribution

Audio recordings are sourced from [Xeno-canto](https://xeno-canto.org/) under [Creative Commons](https://creativecommons.org/licenses/) licences (individual recording licences vary — CC BY, CC BY-NC, CC BY-NC-SA). Please refer to the Xeno-canto recording page for each file's specific licence.

Species taxonomy and class IDs follow the **CUB-200-2011** dataset:
> Wah et al., *"The Caltech-UCSD Birds-200-2011 Dataset"*, CNS-TR-2011-001, California Institute of Technology, 2011.

---

## Citation

If you use this dataset in your research, please cite the original multimodal study and the CUB-200-2011 benchmark:

```bibtex
@article{naranchimeg2018crossdomain,
  title   = {Cross-domain Deep Feature Combination for Bird Species Classification with Audio-visual Data},
  author  = {Naranchimeg, Bold and Zhang, Chao and Akashi, Takuya},
  journal = {arXiv preprint arXiv:1811.10199},
  year    = {2018}
}

@techreport{wah2011cub,
  title       = {The Caltech-UCSD Birds-200-2011 Dataset},
  author      = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
  institution = {California Institute of Technology},
  number      = {CNS-TR-2011-001},
  year        = {2011}
}
```