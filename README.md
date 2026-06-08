# 🏈 NFL Play-Calling Predictor: Run vs. Pass (DSC 148)

**🌟 <a href="https://cadenpascual-nfl-play-predictor-demoapp-dfolai.streamlit.app/" target="_blank" rel="noopener noreferrer">Try the Interactive Web Demo Here!</a>** 📄 **<a href="./docs/Final_Report.pdf" target="_blank" rel="noopener noreferrer">Read our Full ACM Final Report Here</a>** *(Note: Update this path to where your PDF is saved)*

## 📌 Project Overview
This project applies data mining and machine learning techniques to predict NFL offensive play calls (Run vs. Pass). Using modern NFL play-by-play data, we aim to uncover the underlying patterns in offensive play-calling behavior across different game situations. 

This repository contains the codebase, interactive web application, and documentation for our **DSC 148** final project. For a full breakdown of our methodology, literature review, and comprehensive results (including ablation and case studies), please refer to our full report linked above.

## 📂 Repository Structure

```text
nfl-play-predictor/
│
├── docs/                        # ACM Format Final Report
│   └── Final_Report.pdf
│
├── data/                        # Data folder (ignored in git)
│   ├── play_by_play_2022.parquet
│   ├── play_by_play_2023.parquet
│   ├── play_by_play_2024.parquet
│   └── play_by_play_2025.parquet
│
├── notebooks/                   # Jupyter Notebooks for execution
│   ├── 01_EDA.ipynb             # Data cleaning & Exploratory Data Analysis
│   ├── 02_Modeling.ipynb        # Baseline, Advanced Models & Hyperparameter Tuning
│   └── 03_Evaluation.ipynb      # Final testing, Ablation, and Case Studies
│
├── models/                      # Pre-trained model artifacts
│   └── lgbm_runpass.joblib      # Serialized LightGBM model
│
├── src/                         # Source code modules
│   └── features.py              # Feature engineering pipeline
│
├── demo/                        # Streamlit interactive demo application
│   └── app.py                   # Streamlit app for predictions
│
├── README.md                    # Project documentation
└── requirements.txt             # Required Python packages
```

## 📊 The Dataset & Task
We utilized historical NFL play-by-play data sourced via <a href="https://www.nflfastr.com/" target="_blank" rel="noopener noreferrer">nflfastR</a> / <a href="https://www.kaggle.com/datasets/nflverse/nfl-play-by-play-data" target="_blank" rel="noopener noreferrer">Kaggle</a>. The dataset contains over 150,000 clean run/pass plays from the 2022-2024 seasons for training, and we utilized the 2025 season as an unseen testing set to prevent data leakage. 

Our primary predictive task is a binary classification problem: **Given a pre-snap game situation, predict whether the offensive team will execute a RUN or PASS play.** We evaluated our LightGBM and baseline Logistic Regression models using Accuracy, Precision, Recall, and F1 Scores. 

## 🎮 Interactive Streamlit Demo
You can test our predictive models in real-time directly in your browser:
👉 **<a href="https://cadenpascual-nfl-play-predictor-demoapp-dfolai.streamlit.app/" target="_blank" rel="noopener noreferrer">Launch the NFL Play Predictor</a>**

The application provides two prediction points:
- **Before Lineup (Default):** Uses only pre-snap game state (down, distance, score, clock, field position, Vegas market info).
- **After Lineup:** Additionally includes pre-snap formation cues (shotgun, no-huddle) for more accurate predictions.

### 💻 Setup & Installation (Running Locally)
To run this project and the interactive web demo locally on your machine:

```bash
# Clone the repository
git clone [https://github.com/cadenpascual/nfl-play-predictor.git](https://github.com/cadenpascual/nfl-play-predictor.git)
cd nfl-play-predictor

# Install dependencies 
pip install -r requirements.txt

# Run the Streamlit web app
streamlit run demo/app.py
```
*The app will automatically open in your default web browser at `http://localhost:8501`.*

## 👥 Authors
Caden Pascual - DSC 148 Final Project, University of California, San Diego (UCSD)

## 📚 References
- NFL play-by-play data: <a href="https://www.nflfastr.com/" target="_blank" rel="noopener noreferrer">nflfastR</a> / <a href="https://www.kaggle.com/datasets/nflverse/nfl-play-by-play-data" target="_blank" rel="noopener noreferrer">Kaggle</a>
- LightGBM: <a href="https://lightgbm.readthedocs.io/" target="_blank" rel="noopener noreferrer">Documentation</a>
- Streamlit: <a href="https://streamlit.io/docs" target="_blank" rel="noopener noreferrer">Documentation</a>
