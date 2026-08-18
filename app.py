import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('dark_background')
sns.set_style('darkgrid') 

# Set up page layout
st.set_page_config(page_title="Model Inference Dashboard", layout="wide")
st.title("🔋 NASA Battery Diagnostic Model")

# 1. Sidebar Configuration: Load Saved Models
st.sidebar.header("📁 Model Configuration")

# Define a dictionary of your saved .pkl models (Update paths or file names as needed)
AVAILABLE_MODELS = {
    "Random Forest": "model/random_forest.pkl",
    "Logistic Regression": ["model/logistic_regression.pkl", "model/logistic_regression_scaler.pkl"],
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": ["model/k_nearest_neighbors.pkl","model/k_nearest_neighbors_scaler.pkl"],
    "Naive Bayes": ["model/gaussian_naive_bayes.pkl","model/gaussian_naive_bayes_scaler.pkl"]
}

MODEL_WITH_SCALER = ["Logistic Regression", "KNN", "Naive Bayes"]

# Allow selection of multiple models
selected_model_names = st.sidebar.multiselect(
    "Select models to evaluate:", 
    options=list(AVAILABLE_MODELS.keys())
)

# 2. Main Panel: File Upload
uploaded_file = st.file_uploader("Upload your test_data.csv file", type=["csv"])

if uploaded_file is not None:
    # Read csv data
    df = pd.read_csv(uploaded_file)

    # Drop the columns
    col_to_drop = ["time_to_10soc",
    "time_to_25soc", 
    "time_to_40soc", 
    "time_to_50soc",
    "curr_min"]

    df.drop(col_to_drop, axis=1, inplace=True)
    
    st.write("## 📋 Test Data Preview", df.head())

    # Default list to store metrics value
    Accuracy = []
    Auc = []
    Precision = []
    Recall = []
    F1 = []
    MCC = []

    # 3. Model Execution
    if selected_model_names:
        if st.button("🚀 Run Inference"):
            # Create a copy of the dataframe to store predictions from different models
            results_df = df.copy()
            
            for model_name in selected_model_names:
                model_value = AVAILABLE_MODELS[model_name]
                if model_name in MODEL_WITH_SCALER:
                    model_path = model_value[0]
                else:
                    model_path = model_value

                # Check if the model file exists locally
                if os.path.exists(model_path):
                    try:
                        # Load the model using joblib
                        loaded_model = joblib.load(model_path)
                        
                        # Preprocess the df
                        X = df.drop("degradation", axis=1)
                        if model_name in MODEL_WITH_SCALER:
                            scaler = joblib.load(model_value[1])
                            X = scaler.transform(X)
                        y = df['degradation']
                        predictions = loaded_model.predict(X)
                        probabilities = loaded_model.predict_proba(X)

                        # Store Metric in list
                        Accuracy.append(accuracy_score(y, predictions))
                        Auc.append(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"))
                        Precision.append(precision_score(y, predictions,average="macro"))
                        Recall.append(recall_score(y, predictions,average="macro"))
                        F1.append(f1_score(y, predictions,average="macro"))
                        MCC.append(matthews_corrcoef(y, predictions))

                        # Add predictions as a new column
                        results_df[f"{model_name} Predictions"] = predictions
                        st.success(f"Successfully ran inference for **{model_name}**.")

                        st.write(f'## Report: {model_name}')
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🧩 Confusion Matrix")
                            labels = sorted(list(set(y).union(set(predictions))))
                            cm = confusion_matrix(y, predictions, labels=labels)
                            
                            fig, ax = plt.subplots(figsize=(5, 4))
                            sns.heatmap(
                                cm, 
                                annot=True, 
                                fmt="d", 
                                cmap="rocket", 
                                xticklabels=labels, 
                                yticklabels=labels, 
                                cbar=False,
                                ax=ax
                            )
                            ax.set_ylabel('Actual Label')
                            ax.set_xlabel('Predicted Label')
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            st.subheader("📋 Classification Report")
                            
                            # Generate the report as a dictionary, then convert to DataFrame
                            report_dict = classification_report(y, predictions, output_dict=True)
                            report_df = pd.DataFrame(report_dict).transpose()
                            
                            # Format values as percentages/decimals and display nicely
                            st.dataframe(report_df.style.format(precision=2))
                        
                    except Exception as e:
                        st.error(f"Error running **{model_name}**: {e}")
                else:
                    st.error(f"Model file not found at: `{model_path}`. Please verify the file path.")
            
            # 4. Display Results
            metrics_df = pd.DataFrame(index=selected_model_names, data={"Accuracy":Accuracy, "AUC": Auc, "Precision": Precision, "Recall": Recall, "F1": F1, "MCC": MCC})

            st.write('## 🎯 Metrics')
            st.dataframe(metrics_df)

            st.write("## 📊 Inference Results")
            st.dataframe(results_df)
            
            # Download capability for the results
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="model_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("💡 Please select at least one model from the sidebar menu.")
        
else:
    st.info("👋 Please upload a CSV file to get started.")
