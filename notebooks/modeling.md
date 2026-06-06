# 🏈 Phase 2: Modeling & Evaluation (Next Steps)

This document outlines the workflow for our `02_Modeling_and_Tuning.ipynb` and `03_Final_Evaluation.ipynb` notebooks. The goal is to predict NFL play calls (`run` vs. `pass`) using our cleaned 2022-2024 training data.

### 🛠️ 1. Data Preparation (Feature Engineering)
Before feeding data into the models, we need to prep the features:
- [ ] **Define X and y:** Separate our target variable (`play_type`) from our features (`down`, `ydstogo`, `score_differential`, etc.).
- [ ] **Encode the Target:** Convert `play_type` into binary numbers (e.g., `pass` = 1, `run` = 0).
- [ ] **Handle Categorical Variables:** If we added any text columns (like `formation`), apply One-Hot Encoding (`pd.get_dummies()`).
- [ ] **Scaling (Optional but recommended):** Scale continuous features like `ydstogo` and `yardline_100` using `StandardScaler` (required for Logistic Regression, but optional for Random Forests).

### 📉 2. The Baseline Model
The rubric requires a baseline model to prove our advanced models are actually doing something useful.
- [ ] **Train a Logistic Regression Model:** This is the standard baseline for binary classification.
- [ ] **Record Baseline Metrics:** Calculate Baseline Accuracy and print a Classification Report (Precision, Recall, F1-score). 
- [ ] *Note: If teams pass 60% of the time, our baseline model must score higher than 60% to be considered successful!*

### 🌲 3. Advanced Modeling 
Train more complex models that handle non-linear sports data better.
- [ ] **Train a Random Forest Classifier:** Excellent out-of-the-box performance for tabular data.
- [ ] **Train an XGBoost / LightGBM Classifier (Optional but highly accurate):** Gradient boosting usually yields the highest accuracy for NFL play-by-play data.

### ⚙️ 4. Hyperparameter Tuning (Parameter Sensitivity)
*Crucial for the "Parameter Sensitivity" rubric section.*
- [ ] **Set up Grid Search / Randomized Search:** Use 5-fold Cross-Validation on the training data.
- [ ] **Tune specific parameters:** Test different values for `max_depth`, `n_estimators`, or `learning_rate`.
- [ ] **Visualize Sensitivity:** Create a line chart plotting the Hyperparameter Value (X-axis) vs. Model Accuracy (Y-axis) to show the professor exactly how tuning improved the model.

### 🚀 5. Final Evaluation (The 2025 Test Set)
*To be done ONLY at the very end in a separate notebook to prevent data leakage.*
- [ ] **Load `play_by_play_2025.parquet`:** Apply the exact same filtering and scaling steps used on the training set.
- [ ] **Predict the Future:** Run our *single best tuned model* on this 2025 test set.
- [ ] **Generate Visuals for the Report:** - Plot a **Confusion Matrix** to show exactly where the model got confused (e.g., guessing pass when it was actually a run).
    - Plot a **Feature Importance Chart** to show the professor which variable (like `ydstogo` or `down`) was the most important factor in the model's brain.