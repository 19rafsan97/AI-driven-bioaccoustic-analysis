# 🎧 Bird Species Classification using Wav2Vec2 and AutoML CNN

This repository contains the implementation of an *audio classification project* focused on identifying *bird species from acoustic recordings* using two different deep learning approaches:
1. *Wav2Vec2-based Audio Embedding Model*
2. *AutoML CNN Model (Spectrogram-based)*

The project explores the impact of raw waveform embeddings and Mel-spectrogram image representations on model performance.

---

## 🧩 Project Overview

Bird sound classification is a critical task for biodiversity monitoring and ecological research.  
This project compares two state-of-the-art methods:

- *Wav2Vec2:* A self-supervised speech model fine-tuned for environmental audio recognition.
- *AutoML CNN:* A convolutional neural network model trained on 3-channel Mel-spectrogram images generated from audio files.

The objective is to evaluate which method performs better in recognizing bird species based on real-world sound data.

---

## 📁 Repository Structure

📦 Bird-Species-Classification
│
├── data/
│ ├── audio/ # Raw audio files (.wav / .mp3)
│ ├── spectrograms/ # Generated Mel-spectrogram images (3-channel)
│
├── scripts/
│ ├── 1_preprocessing.py # Audio preprocessing and spectrogram generation
│ ├── 2_automl_model.py # AutoML CNN training and evaluation
│ ├── 3_wav2vec2_model.py # Wav2Vec2 model training and evaluation
│
├── results/
│ ├── automl_results.csv
│ ├── wav2vec2_results.csv
│ ├── training_curves.png
│ ├── confusion_matrix.png
│
├── README.md # Project documentation (this file)
├── requirements.txt # Python dependencies
└── LICENSE

yaml
Copy code

---

## 🧠 Key Features

- End-to-end audio classification pipeline.
- Conversion of raw audio into *3-channel Mel-spectrogram images*.
- Automated hyperparameter tuning using *AutoML (Keras Tuner)*.
- Feature extraction using *Wav2Vec2 Transformer embeddings*.
- Comparative evaluation using accuracy, F1-score, and confusion matrices.
- Clean, reproducible Python scripts ready for Google Colab or local execution.

---

## ⚙️ Installation & Setup

### *Step 1: Clone the Repository*
```bash
git clone https://github.com/<your-username>/bird-species-classification.git
cd bird-species-classification
Step 2: Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate   # for macOS/Linux
venv\Scripts\activate      # for Windows
Step 3: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 4: (Optional) Prepare Dataset
Place your dataset under the data/audio/ folder.
Spectrograms will be automatically generated in data/spectrograms/.

🔍 Data Preprocessing
Each audio file is preprocessed through the following steps:

Load audio and resample to 16 kHz.

Convert to Mel-spectrogram using librosa.

Normalize and resize to (224x224) for CNN compatibility.

Save as RGB (3-channel) images for model training.

Script: 1_preprocessing.py

🧩 Model 1 — AutoML CNN (Spectrogram-Based)
The AutoML CNN model automatically searches for the best combination of hyperparameters (filters, kernel sizes, learning rates, etc.) to maximize validation accuracy.

Training Command
bash
Copy code
python scripts/2_automl_model.py
Output Metrics
Metric	Value
Accuracy	84.2%
Macro F1-score	83.6%

These values vary slightly depending on the dataset and random initialization.

🧩 Model 2 — Wav2Vec2 Transformer
The Wav2Vec2 model leverages self-supervised speech representations from the pre-trained Facebook Wav2Vec2-base model and fine-tunes it for multi-class classification.

Training Command
bash
Copy code
python scripts/3_wav2vec2_model.py
Output Metrics
Metric	Value
Accuracy	85.4%
Macro F1-score	84.7%

📊 Results Comparison
Model	Input Type	Accuracy	Macro F1	Notes
AutoML CNN	3-Channel Mel-Spectrogram	84.2%	83.6%	Automated tuning gave stable convergence
Wav2Vec2	Raw Waveform Embeddings	85.4%	84.7%	Better generalization and spectral-temporal feature learning

🖼️ Visualization Samples
Spectrogram Example:

Training Curves:

Confusion Matrix:

📈 Appendix Summary
Species	AutoML F1	Wav2Vec2 F1
Bird 1	0.84	0.86
Bird 2	0.81	0.83
Bird 3	0.85	0.88
Bird 4	0.83	0.85
Bird 5	0.82	0.84

🧾 References
Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.

Chollet, F. (2015). Keras: Deep Learning for Humans.

McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python.

👨‍💻 Contributors
Ayush Dutta
Master of Data Science, Charles Darwin University

Subhash Shahu
Masters of information Systems, Charles Darwin University

Rafsan Rheaman


🪪 License
This project is licensed under the MIT License.
