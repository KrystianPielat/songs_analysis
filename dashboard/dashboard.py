import streamlit as st
import pandas as pd
import os
from PIL import Image

# Function to load evaluation results
def load_evaluation_table(folder):
    """Loads the evaluation results from the CSV file in the selected folder."""
    evaluation_file = os.path.join(folder, 'evaluation_results.csv')
    if os.path.exists(evaluation_file):
        df = pd.read_csv(evaluation_file)
        return df
    else:
        st.error("Evaluation file not found.")
        return None

# Function to load SHAP plots
def load_shap_plot(folder, plot_name):
    """Loads a SHAP plot from the selected folder."""
    plot_file = os.path.join(folder, plot_name)
    if os.path.exists(plot_file):
        img = Image.open(plot_file)
        return img
    else:
        st.error(f"{plot_name} not found.")
        return None

# Get list of available result folders
def get_result_folders(base_dir="results"):
    """Retrieves the list of result directories."""
    if os.path.exists(base_dir):
        folders = [f.path for f in os.scandir(base_dir) if f.is_dir() and '.ipynb' not in f.path]
        return folders
    else:
        st.error(f"Base directory {base_dir} not found.")
        return []

# Streamlit app starts here
st.title("Results Dashboard")

# Dropdown to select the folder with results
result_folders = get_result_folders()  # Assuming 'results' is the base directory
selected_folder = st.selectbox("Select a folder with results", result_folders)

if selected_folder:
    st.subheader(f"Results from: {selected_folder}")
    
    # Display the evaluation table
    st.subheader("Evaluation Metrics")
    evaluation_df = load_evaluation_table(selected_folder)
    if evaluation_df is not None:
        st.dataframe(evaluation_df)
    
    # Display the SHAP summary plot
    st.subheader("SHAP Summary Plot")
    summary_plot = load_shap_plot(selected_folder, 'shap_summary_plot.png')
    if summary_plot is not None:
        st.image(summary_plot, caption="SHAP Summary Plot")

    # Display the SHAP beeswarm plot
    st.subheader("SHAP Beeswarm Plot")
    beeswarm_plot = load_shap_plot(selected_folder, 'shap_beeswarm_plot.png')
    if beeswarm_plot is not None:
        st.image(beeswarm_plot, caption="SHAP Beeswarm Plot")