# Multi-Modal Violence and Anomaly Detection (Audio Module)

This repository contains the **Audio-Based Threat Detection Module** for the final year research project: *Multi-Modal Violence and Anomaly Detection in Public Surveillance Using Deep Learning and Temporal Models*.

## Project Overview
This module specializes in identifying and categorizing violent events (Fighting, Shooting, Stabbing, Explosions, Harassment, Theft) through synchronized audio-visual data. It employs state-of-the-art acoustic feature extraction and temporal modeling to provide robust real-time detections.

## Key Features
- **Acoustic Intelligence:** Utilizes **PANNs (Pretrained Audio Neural Networks)** for robust audio feature extraction.
- **Temporal Modeling:** Implements **Temporal Transformers** to capture the evolution of violent events over time.
- **Explainable AI (XAI):** Uses SHAP/LIME to provide interpretable heatmaps and human-readable reasoning for alerts.
- **Fuzzy Logic:** Integrates fuzzy reasoning to assess violence levels (e.g., distinguishing between mild aggression and high-threat attacks).

## Tech Stack
- **Framework:** PyTorch / Torchaudio
- **Audio Processing:** Librosa, PANNs (CNN14)
- **Interpretability:** SHAP, LIME
- **Logic Handling:** Scikit-Fuzzy
- **Training:** Optimized for Google Colab with Google Drive integration.

## Dataset
- **XD-Violence:** Primary dataset containing ~217 hours of synchronized audio-visual clips.
- **UCF-Crime:** Supplementary dataset for general anomaly detection.

## Structure
- `configs/`: YAML configuration files.
- `data_preprocessing/`: Scripts for audio extraction and spectrogram generation.
- `models/`: Implementation of PANNs and Temporal Transformers.
- `training/`: Training loops and validation logic.
- `utils/`: Helper functions for logging and visualization.

---
**Group Paragon** | Faculty of Information Technology, University of Moratuwa | 2026
