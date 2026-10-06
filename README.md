# Peruvian Sign Language (LSP) Translator - Bidirectional GRU

Real-time recognition of Peruvian Sign Language using MediaPipe hand/pose landmarks and a bidirectional GRU classifier, including the tooling to collect and augment your own dataset.

Versión en español: [README.es.md](README.es.md)

> Academic project.

## Problem

Peruvian Sign Language (Lengua de Señas Peruana, LSP) has very little public data or tooling. This project provides an end-to-end pipeline - collect landmarks from a webcam, augment them, train a sequence model and translate signs live - so a dataset can be built from scratch.

## Approach

```mermaid
flowchart LR
  A[Webcam] --> B[MediaPipe hands + pose]
  B --> C[Feature extraction<br/>60-frame sequences]
  C --> D[Data augmentation]
  D --> E[Bidirectional GRU]
  E --> F[Sign prediction + confidence]
```

- **Features:** 60-frame sequences. Each frame has 157 features: 126 hand landmark values (21 landmarks x 3 coordinates x 2 hands), 24 pose values and 7 motion/velocity values (`src/data_collection/feature_extractor.py`).
- **Model:** LayerNorm -> 2 x Bidirectional GRU (128 units, dropout, L2) -> optional attention pooling -> Dense(128) -> Softmax (`src/training/model_builder.py`). Adam optimizer, early stopping, learning-rate reduction on plateau.
- **Augmentation:** speed changes, pauses, small rotation/scale/translation, Gaussian noise and jitter, left/right hand swap. The technique mix depends on sign type (static letters, dynamic letters, words, phrases).
- **Vocabulary:** 42 signs configured: 29 letters (including J, Z, Ñ, RR, LL), 9 words and 4 phrases (`src/data_collection/sign_config.py`).

## Tech stack

Python 3.11+, TensorFlow/Keras, MediaPipe, OpenCV, NumPy, SciPy, scikit-learn, pandas.

## Getting started

```bash
git clone https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU.git
cd Traductor-Lenguaje-Senas-GRU
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

`run.py` opens a text menu that checks and downloads the MediaPipe model files if missing, and gives access to:

1. Data collection (webcam, per-sign progress dashboard, augmentation, statistics)
2. Model training
3. Model evaluation
4. Real-time translation

A webcam is required for collection and inference.

## Project structure

```
run.py                      # main menu
src/
  data_collection/          # collector, MediaPipe manager, features, motion analysis, augmentation, UI
  training/                 # data loader, model builder, training pipeline
  evaluation/               # model evaluation
  inference/                # real-time translator
  utils/                    # MediaPipe model downloader
docs/                       # architecture, installation, user guide, augmentation guide (Spanish)
tests/                      # test scripts
```

## Status

The pipeline code is complete, but no trained model, dataset or accuracy results are published in this repository. Accuracy figures that appear in the Spanish documentation are illustrative examples, not measurements.

## Credits

- Author: Jhamil Brijan Peña Cárdenas ([@Jaed69](https://github.com/Jaed69))
- Built on MediaPipe and TensorFlow.

## License

No license file is included in this repository.
