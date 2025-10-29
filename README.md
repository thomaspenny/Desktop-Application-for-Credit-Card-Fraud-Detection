# Machine Learning Model Analysis and Interpretability Toolkit

## Project Overview

This application is a comprehensive **Machine Learning Model Analysis and Interpretability Toolkit** developed as the final project for the HCS522 module at Abertay University by Thomas Penny.

The tool provides a graphical user interface (GUI) built with **Tkinter** for loading a dataset, comparing multiple classification models, and generating visual reports on model performance (AUC-ROC, Confusion Matrices, Timings) and interpretability (**SHAP and LIME**).

This project focuses on providing a detailed, explainable comparison of various models in a single, user-friendly environment.

## Key Features

* **Model Comparison:** Runs and compares six popular classification models:
    * K-Nearest Neighbors (KNN)
    * XGBoost
    * Random Forest
    * Logistic Regression
    * Decision Tree
    * Multi-Layer Perceptron (MLP)

* **Data Handling:** Supports loading data from CSV files and includes **automated handling for data imbalance** using **SMOTE, Random Over Sampling, and Random Under Sampling**.

* **Performance Metrics:** Generates immediate visual reports (Matplotlib/Seaborn) for:
    * **AUC-ROC Curves:** For visual comparison of classifier performance.
    * **Confusion Matrices:** To evaluate classification results.
    * **Recall & Precision Scores:** Displayed in a comparative table.
    * **Processing Timings:** Bar chart visualization of model training/testing times.

* **Model Interpretability (XAI):**
    * **SHAP (SHapley Additive exPlanations):** Generates global and local feature importance plots to explain model output.
    * **LIME (Local Interpretable Model-agnostic Explanations):** Provides local explanations for a selected instance/prediction.

* **GUI Interface:** All functionality is accessed via a clean, integrated Tkinter application window.

## Installation and Setup

### Prerequisites

The application requires Python 3.x and several external libraries.

1.  **Clone the Repository (If Applicable):**
    ```bash
    git clone [repository-url]
    cd ML_model_app
    ```

2.  **Install Dependencies:**
    It is highly recommended to use a virtual environment.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap lime
    ```
    *Note: The `shap` library may require additional build dependencies depending on your operating system.*

### Running the Application

Execute the Python script directly:
```bash
python ML_model_app.py
