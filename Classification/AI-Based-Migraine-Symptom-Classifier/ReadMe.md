# 🧠 AI-Based Migraine Symptom Classifier for Triage

This project develops, evaluates, and deploys a machine learning pipeline to classify seven distinct migraine subtypes based on patient-reported symptoms. The goal is to provide physicians with an accurate and highly interpretable AI assistant for early triage and decision support.

## 📂 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `migraine.ipynb` | **Full Academic Document:** Comprehensive workflow covering EDA, SMOTE, **GridSearchCV** hyperparameter tuning, model comparison (LR, RF, XGBoost), and detailed **SHAP Interpretability**. |
| `app.py` | **Streamlit Web Application:** The live prediction interface featuring user input, SHAP local explanation, and Gemini AI reasoning. |
| `best_rf.pkl` | **Deployment Model:** The serialized Random Forest pipeline, including all preprocessing steps (scaler, encoder, model). |
| `label_encoder.pkl` | **Target Variable Key:** Used to ensure accurate mapping of numerical predictions back to human-readable migraine names. |
| `README.md` | This project documentation file. |

---

## 🧪 Modeling Summary & Rigor

### 🛠️ Pipeline & Preprocessing

The pipeline was built using `imblearn.pipeline.Pipeline` to maintain data integrity:

* **Imbalance Handling:** **SMOTE** was applied *only* on the training data within the pipeline to prevent data leakage and ensure balanced classes.
* **Preprocessing:** Numerical features (`Age`, `Duration`, `Frequency`) were scaled using `StandardScaler`. Categorical features (`Location`, `Character`) were handled via `OneHotEncoder`.

### 🏆 Model Comparison (Test Set Metrics)

| Model | Accuracy | Weighted F1 | CV Mean F1 | Std Dev |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 91.25% | 92.45% | 0.8628 | ±0.0311 |
| **Random Forest** | **95.00%** | **95.24%** | 0.8698 | **±0.0229** |
| XGBoost | 90.00% | 90.07% | 0.8803 | ±0.0391 |

**Decision:** The **Random Forest** model was selected for deployment due to its superior **Test Set F1-score (95.24%)** and its lower cross-validation standard deviation, indicating the most reliable and stable generalization performance.

---

## 📊 Explainability and AI Integration

The application employs a dual-layer interpretability strategy:

1.  **SHAP (Shapley Additive Explanations):** Used for mathematical rigor.
    * The app displays the **Top 5 Feature Contributions** (SHAP Force Plot data) for *each* individual patient prediction.
    * This answers: "Which specific symptoms (e.g., Phonophobia, Vomit) pushed the prediction toward this result?"

2.  **Gemini AI Reasoning (LLM):** Used for clinical synthesis.
    * The model prediction, confidence, and the list of SHAP-derived top features are passed to the **Gemini 2.5 Flash API**.
    * The LLM generates a concise, clinically focused summary and triage recommendation, translating the raw model output into **actionable insight**.

---

## 🚀 Run Locally

This application requires Python 3.8+ and a **Gemini API Key** to run the AI Reasoning feature.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Up Gemini API Key:**
    * Obtain your key from Google AI Studio.
    * Create a file named **`.env`** in the root directory of the project.
    * Add your key to the `.env` file in the following format:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

3.  **Run Application:**
    ```bash
    streamlit run app_final.py
    ```

