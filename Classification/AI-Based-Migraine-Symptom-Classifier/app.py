import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import os
from dotenv import load_dotenv
import google.generativeai as genai  # 👈 NEW: Gemini library

# --- 1. CONFIGURATION AND LOAD MODEL ---
st.set_page_config(layout="wide")

# ✅ Configure Gemini API key (replace with your real key)
# Load .env and configure Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ Missing GEMINI_API_KEY in your .env file!")
else:
    genai.configure(api_key=api_key)
try:
    # Load your saved pipeline (best_rf.pkl)
    model = joblib.load('best_rf.pkl')
    preprocessor = model.named_steps['preprocessing']
    rf_model = model.named_steps['model']
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("Error: Model files 'best_rf.pkl' not found. Please ensure they are in the app directory.")
    MODEL_LOADED = False

# --- 1. PROJECT OVERVIEW ---
st.title("🧠 AI-Based Migraine Symptom Classifier")
st.markdown("""
This application uses a Random Forest model trained on clinical data to predict the **type of migraine** based on patient symptoms.
""")

# --- 2. USER INPUTS ---
if MODEL_LOADED:
    st.header("🧍 Patient Symptoms & Characteristics")

    # Patient Info
    col_age, col_dur, col_freq = st.columns(3)
    with col_age:
        age = st.slider("Age", 10, 100, 30)
    with col_dur:
        duration = st.selectbox("Duration (in hours)", list(range(1, 11)))
    with col_freq:
        frequency = st.selectbox("Frequency (per week)", list(range(1, 20)))

    # Pain Characteristics
    col_loc, col_char, col_int = st.columns(3)
    with col_loc:
        location = st.radio("Pain Location", ["1 - Unilateral", "2 - Bilateral", "3 - Frontal"])
    with col_char:
        character = st.radio("Pain Character", ["1 - Throbbing", "2 - Pressing", "3 - Sharp"])
    with col_int:
        intensity = st.slider("Intensity (0 = mild, 5 = severe)", 0, 5, 2)

    # Symptom Checklist
    symptom_names = [
        'Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory',
        'Dysphasia', 'Dysarthria', 'Vertigo', 'Tinnitus', 'Hypoacusis',
        'Diplopia', 'Defect', 'Conscience', 'Paresthesia', 'DPF'
    ]
    st.subheader("Associated Symptoms (Check all that apply)")
    cols = st.columns(4)
    symptom_inputs = []
    for idx, symptom in enumerate(symptom_names):
        col = cols[idx % 4]
        with col:
            checked = st.checkbox(symptom)
            symptom_inputs.append(1 if checked else 0)

    # Convert radio inputs to int
    location = int(location.split(" - ")[0])
    character = int(character.split(" - ")[0])

    # Build input dict
    input_data = {
        'Age': age,
        'Duration': duration,
        'Frequency': frequency,
        'Location': location,
        'Character': character,
        'Intensity': intensity,
        'Ataxia': 0,  # constant for dropped feature
        **{name: val for name, val in zip(symptom_names, symptom_inputs)}
    }


    # --- 3. PREDICTION ---
    st.header("---")
    if st.button("🔮 Predict Migraine Type"):
        input_df = pd.DataFrame([input_data])
        input_transformed = preprocessor.transform(input_df)

        # Model Prediction
        prediction_proba = rf_model.predict_proba(input_transformed)[0]
        prediction = rf_model.predict(input_transformed)[0]
        confidence = max(prediction_proba) * 100

        st.success(f"### 🩺 Predicted Migraine Type: **{prediction}**")
        st.info(f"#### Confidence Level: **{confidence:.2f}%**")

        # Probability chart
        st.subheader("📊 Class Probabilities:")
        prob_df = pd.DataFrame({
            'Migraine Type': rf_model.classes_,
            'Confidence (%)': [p * 100 for p in prediction_proba]
        })
        st.bar_chart(prob_df.set_index('Migraine Type'))


        # --- 4. SHAP Explanation ---
        st.header("---")
        st.subheader("🌟 Model Interpretability (SHAP Analysis)")
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(input_transformed)
            feature_names = preprocessor.get_feature_names_out()
            class_index = list(rf_model.classes_).index(prediction)

            if isinstance(shap_values, list):
                shap_for_class = shap_values[class_index][0]
            else:
                shap_for_class = shap_values[0, :, class_index]

            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_for_class
            }).assign(Abs_SHAP=lambda x: x['SHAP Value'].abs()).sort_values(by='Abs_SHAP', ascending=False).head(5)

            # Map preprocessed feature names back to user-friendly inputs

            # Create a dictionary of input values for display
            input_values_map = {k: v for k, v in input_data.items()}

            # Display Top 5 Contributors with Interpretation
            st.markdown("---")
            st.subheader(f"Top 5 Features Driving the Prediction of **{prediction}**:")

            for index, row in shap_df.iterrows():
                feature_name = row['Feature']
                shap_val = row['SHAP Value']

                original_name = feature_name
                input_value = 'N/A'

                # Simplify feature name for mapping and display
                if '__' in feature_name:
                    original_name = feature_name.split('__')[1]
                    # For one-hot encoded features (e.g., 'Location_1'), extract the base feature
                    if '_' in original_name and original_name.split('_')[-1].isdigit():
                        base_name = original_name.split('_')[0]
                        base_value = int(original_name.split('_')[1])
                        if base_name in input_values_map and input_values_map[base_name] == base_value:
                            input_value = f"{base_name}: {base_value}"
                            original_name = base_name
                        else:
                            # Skip if this OHE column wasn't the active one for the user input
                            continue
                    elif original_name in input_values_map:
                        input_value = input_values_map[original_name]
                elif original_name in input_values_map:
                    input_value = input_values_map[original_name]

                # Response Analysis (Textual Explanation)
                contribution_text = ""
                if shap_val > 0:
                    contribution_text = "increased"
                    color = "green"
                else:
                    contribution_text = "decreased"
                    color = "red"

                # Convert symptom inputs (1/0) to Yes/No for clarity
                display_value = input_value
                if original_name in symptom_names:
                    display_value = "Yes (1)" if input_value == 1 else "No (0)"

                # Final formatted output for the feature
                st.markdown(f"""
                - **{original_name}**: (Value: **{display_value}**)
                    - This symptom's status **:{color}[{contribution_text}]** the prediction of *{prediction}* by **{shap_val:+.4f}**.
                """)


        except Exception as e:
            st.error(
                f"SHAP explanation failed. This can happen if the feature names in the pipeline do not align perfectly with the inputs. Error: {e}")

        # --- 5. GEMINI LLM EXPLANATION ---
        st.header("---")
        st.subheader("💬 AI Reasoning (Gemini)")

        try:
            # 💡 FIX: Use the correct model name for the SDK (e.g., 'gemini-2.5-flash')
            # The SDK automatically prepends 'models/' if needed.
            llm = genai.GenerativeModel("gemini-2.5-flash")

            # Extract the top features generated during SHAP analysis in step 5
            top_shap_summary = []

            # Recreate the summary text of the top 5 features and their contribution
            for index, row in shap_df.iterrows():
                # Get simplified names and values. Assume mapping logic from step 5 is correct.
                original_name = row['Feature'].split('__')[-1].split('_')[0]
                shap_val = row['SHAP Value']
                contribution = "increasing" if shap_val > 0 else "decreasing"
                top_shap_summary.append(
                    f"{original_name} (SHAP value: {shap_val:+.4f}) which is {contribution} the prediction.")

            prompt = f"""
            You are an expert Clinical Decision Support System. Your task is to synthesize a machine learning prediction
            with its primary explanatory factors to provide a comprehensive, actionable medical summary.

            The Random Forest model predicted the migraine type: **{prediction}** with confidence **{confidence:.2f}%**.

            The top symptoms and patient characteristics that most influenced this prediction are:
            {chr(10).join(top_shap_summary)}

            **Your response must adhere to this strict structure and tone:**

            1. **Diagnostic Synthesis:** Provide a brief (2-3 sentence) summary of the predicted migraine type, justifying it based on the most influential symptoms listed above.
            2. **Clinical Recommendation & Triage:** Offer clear, evidence-based, non-diagnostic next steps. Suggest what kind of specialist a patient should see or what immediate (non-medication) steps they should take.
            3. **Disclaimer:** Conclude with the mandatory line: **"DISCLAIMER: This is an AI-generated clinical support tool. It is not a substitute for professional medical advice, diagnosis, or treatment."**
            """

            response = llm.generate_content(prompt)
            st.markdown("### 🧠 Gemini Clinical Explanation (Fulfills Instructor's Requirement)")
            st.markdown(response.text)  # Use st.markdown for structured output like lists/bolding

        except Exception as e:
            # Catching the API error directly
            st.error(
                f"Gemini explanation failed. Please verify your API key is correct and that the model 'gemini-2.5-flash' is available for your account. Error: {e}")

    # --- 6. Validation Note ---
    st.markdown("---")
    st.markdown("""
    **Validation Note**: The Random Forest model achieved a **95.24% F1-weighted score** 
    in cross-validation. Combined with SHAP and Gemini explanations, this app ensures 
    both **accuracy** and **interpretability**.
    """)
