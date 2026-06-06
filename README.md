# 🏈 NFL Play-Calling Predictor: Run vs. Pass (DSC 148)

## 📌 Project Overview
This project applies data mining and machine learning techniques to predict NFL offensive play calls (Run vs. Pass). Using modern NFL play-by-play data, we aim to uncover the underlying patterns in coaching decisions based on game context (down, distance, score differential, and field position). 

This repository was created as the final project for **DSC 148**, fulfilling all requirements for exploratory data analysis, baseline modeling, advanced predictive modeling, and parameter sensitivity.

## 📂 Repository Structure
To maintain a professional and reproducible workflow, our code and data are separated into distinct directories:

` ` `text
nfl-play-predictor/
│
├── data/                      # Data folder (ignored in git)
│   ├── play_by_play_2022.parquet
│   ├── play_by_play_2023.parquet
│   ├── play_by_play_2024.parquet
│   └── play_by_play_2025.parquet
│
├── notebooks/                 # Jupyter Notebooks for execution
│   ├── 01_EDA.ipynb           # Data cleaning & Exploratory Data Analysis
│   ├── 02_Modeling.ipynb      # Baseline, Advanced Models & Hyperparameter Tuning
│   └── 03_Evaluation.ipynb    # Final testing on unseen 2025 data
│
├── README.md                  # Project documentation
└── requirements.txt           # Required Python packages
` ` `
*(Note: Remove the spaces between the backticks above when pasting)*

## 📊 The Dataset
We utilize historical NFL play-by-play data sourced via `nflfastR` / Kaggle. The data is stored in `.parquet` format for highly efficient columnar reading and strict data-type preservation.

To prevent data leakage and simulate a real-world predictive task, we utilized a **time-based train/test split**:
* **Training Set:** 2022, 2023, and 2024 seasons (~150,000 clean run/pass plays)
* **Testing Set:** 2025 season (Unseen future data for final evaluation)

*Note: Non-action plays such as timeouts, penalties, and quarter breaks were filtered out during preprocessing.*

## 🚀 Workflow & Methodology

### 1. Exploratory Data Analysis (EDA)
In `01_EDA.ipynb`, we explored the training dataset to uncover play-calling tendencies. Key features analyzed include `down`, `ydstogo`, `yardline_100`, and `score_differential`. We generated visualizations (boxplots, bar charts) to highlight the correlation between game states and pass probabilities.

### 2. Modeling & Parameter Sensitivity
In `02_Modeling.ipynb`, we trained two levels of machine learning models:
* **Baseline Model:** Logistic Regression to establish our baseline accuracy.
* **Advanced Model:** Random Forest / Gradient Boosting (XGBoost) to capture non-linear relationships in the sports data.
* **Parameter Sensitivity:** We utilized cross-validation and Grid Search to tune hyperparameters (e.g., `max_depth`, `n_estimators`). Model performance across different parameter states is explicitly visualized in this notebook.

### 3. Final Evaluation
In `03_Evaluation.ipynb`, our best-tuned model is evaluated against the unseen 2025 test dataset. We generate a final Confusion Matrix, calculate Precision/Recall/F1 scores, and plot Feature Importances to determine which game-state variables drove the model's decisions.

## 💻 Setup & Installation
To run this project locally, clone the repository and install the required dependencies:

` ` `bash
# Clone the repo
git clone https://github.com/YourUsername/nfl-play-predictor.git
cd nfl-play-predictor

# Install dependencies
pip install -r requirements.txt
` ` `
*(Note: Remove the spaces between the backticks above when pasting)*