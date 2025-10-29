# ============================================================================
# ============ Thomas Penny, Abertay University 2025, HCS522 =================
# ============================================================================

import threading
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import copy
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import shap
from lime.lime_tabular import LimeTabularExplainer
import webbrowser


# ============================================================================
# VARIABLES
# ============================================================================

# Handle cancellation
cancel_event = threading.Event()
lime_cancel_event = threading.Event()
scaler_lock = threading.Lock()

# Tkinter UI
root = tk.Tk()
root.title("ML Model Simulation")
root.option_add('*Font', 'Tahoma 9')

# menu bar
menubar = tk.Menu(root)
root.config(menu=menubar)

# Global variables for UI
result_box = None
progress_label = None
progress_bar = None
cancel_button = None
run_all_models_button = None
reset_button = None
shap_button_ref = None
fraud_analysis_button_ref = None
selected_model = None
selected_model_name = ""

# Over/under sampling
sampling_method = tk.StringVar(value="None")
model_selection = tk.StringVar(value="Logistic Regression")
shap_model_selection = tk.StringVar(value="Logistic Regression")

# Store results of batch processing
model_results = []

# Store model predictions for AUC ROC
model_predictions = {}

# Batch processing variables
running_all_models = False
current_model_index = 0
all_models_list = []

# Train-test split settings
train_test_split_ratio = tk.DoubleVar(value=0.8)
random_seed = tk.IntVar(value=2025)
df = None
X = None
y = None
data_loaded = False

# Font constants
font_family = 'Tahoma'
font_s = (font_family, 9)   
font_s_b = (font_family, 9, 'bold')
font_m = (font_family, 10)          
font_m_b = (font_family, 10, 'bold') 
font_l = (font_family, 11)          
font_l_b = (font_family, 11, 'bold') 
font_xl_b = (font_family, 12, 'bold') 
font_code = ('Consolas', 9)         

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*_check_feature_names.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*_check_n_features.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*BaseEstimator.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*_validate_data.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ensure_all_finite.*", category=FutureWarning)

# ============================================================================
# CSV IMPORT
# ============================================================================

# Load CSV file using file dialog
def load_csv_file():
    global df, X, y, data_loaded
        
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Credit Card Dataset",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        return False
    
    # Load the CSV file
    result_box.insert(tk.END, f"Loading dataset from: {file_path}\n")
    root.update()
    df = pd.read_csv(file_path)

    # Validate the dataset structure
    if 'Class' not in df.columns:
        messagebox.showerror("Invalid Dataset", 
                           "Dataset must contain a 'Class' column for fraud labels.")
        return False
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    pd.set_option('future.no_silent_downcasting', True)

    # Dataset info
    fraud_count = (y == 1).sum()
    normal_count = (y == 0).sum()
    total_count = len(df)

    # Show dataset info
    result_box.insert(tk.END, f"Dataset loaded successfully\n")
    result_box.insert(tk.END, f"Total records: {total_count:,}\n")
    result_box.insert(tk.END, f"Features: {len(X.columns)}\n")
    result_box.insert(tk.END, f"Normal transactions: {normal_count:,} ({normal_count/total_count*100:.1f}%)\n")
    result_box.insert(tk.END, f"Fraudulent transactions: {fraud_count:,} ({fraud_count/total_count*100:.1f}%)\n")
    result_box.insert(tk.END, f"Feature columns: {', '.join(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}\n\n")
    result_box.insert(tk.END, "Dataset ready for analysis\n\n")
    data_loaded = True

    enable_data_dependent_ui()
    return True

# Enable UI elements that require loaded data
def enable_data_dependent_ui():
    run_button.config(state=tk.NORMAL)
    run_all_models_button.config(state=tk.NORMAL)
    show_table_button.config(state=tk.NORMAL)
    shap_button_ref.config(state=tk.NORMAL)
    fraud_analysis_button_ref.config(state=tk.NORMAL)

    # Update main UI status label for dataset load
    global status_label
    status_label.config(text="Dataset loaded", fg='green')

    progress_label.config(text="Dataset loaded - Ready to run models")

# Enable UI elements that require batch processing results
def enable_results_dependent_ui():
    if model_results: 
        visualize_button.config(state=tk.NORMAL)

# Disable UI elements that require batch processing results
def disable_results_dependent_ui():
    visualize_button.config(state=tk.DISABLED)

# Disable UI elements that require loaded data
def disable_data_dependent_ui():
    run_button.config(state=tk.DISABLED)
    run_all_models_button.config(state=tk.DISABLED)
    show_table_button.config(state=tk.DISABLED)
    visualize_button.config(state=tk.DISABLED)
    shap_button_ref.config(state=tk.DISABLED)
    fraud_analysis_button_ref.config(state=tk.DISABLED)



# ============================================================================
# SINGLE MODEL
# ============================================================================

# Run model in thread
def run_simulation_thread(model, model_name):
    # Use prepared global variables from run_selected_model
    global X_train_std, X_test_std, y_train, y_test
    global reset_button
    
    # Reset cancellation event and UI
    cancel_event.clear()
    
    # Disable buttons during simulation
    cancel_button.config(state=tk.NORMAL)
    if reset_button:
        reset_button.config(state=tk.DISABLED)
    
    result_box.delete(1.0, tk.END)
    progress_label.config(text=f"Running {model_name} simulation, please wait...")
    progress_bar.config(mode='indeterminate')
    progress_bar.start()
    root.update()

    # Apply sampling based on selected method
    sampling_method_value = sampling_method.get()
    
    # Show original class distribution
    original_class_0 = (y_train == 0).sum()
    original_class_1 = (y_train == 1).sum()
    
    if sampling_method_value == "SMOTE":
        result_box.insert(tk.END, "SMOTE applied to training data.\n")
        sm = SMOTE(random_state=random_seed.get())
        X_train_std, y_train = sm.fit_resample(X_train_std, y_train)
    elif sampling_method_value == "RandomOverSampler":
        result_box.insert(tk.END, "RandomOverSampler applied to training data.\n")
        ros = RandomOverSampler(random_state=random_seed.get())
        X_train_std, y_train = ros.fit_resample(X_train_std, y_train)
    elif sampling_method_value == "RandomUnderSampler":
        result_box.insert(tk.END, "RandomUnderSampler applied to training data.\n")
        rus = RandomUnderSampler(random_state=random_seed.get())
        X_train_std, y_train = rus.fit_resample(X_train_std, y_train)
    else:
        result_box.insert(tk.END, "No sampling method applied.\n")
    
    # Show class balance ratios
    if sampling_method_value != "None":
        result_box.insert(tk.END, f"Original class balance ratio - Non-fraud:Fraud = {original_class_0:,}:{original_class_1:,}\n")
        new_class_0 = (y_train == 0).sum()
        new_class_1 = (y_train == 1).sum()
        result_box.insert(tk.END, f"New class balance ratio - Non-fraud:Fraud = {new_class_0:,}:{new_class_1:,}\n")
    
    result_box.insert(tk.END, "\n") 
    root.update()

    # Train the model
    result_box.insert(tk.END, f"Training {model_name}...\n")
    root.update()
    model.fit(X_train_std, y_train)
    
    # Real-time prediction display
    rt_process_start = time.perf_counter()
    y_pred_realtime = []
    
    progress_bar.config(mode='determinate', maximum=len(X_test_std))
    
    result_box.insert(tk.END, f"Starting real-time {model_name} predictions for {len(X_test_std)} samples...\n\n")
    root.update()

    for i, row in enumerate(X_test_std):
        if cancel_event.is_set():
            result_box.insert(tk.END, "\nSimulation canceled by user.\n")
            break
        
        start_time = time.perf_counter()
        
        # Reshape for single-row prediction
        row_reshaped = row.reshape(1, -1)
        pred = model.predict(row_reshaped)[0]
        
        end_time = time.perf_counter()
        
        if pred == 1:
            # Use iloc to access by position instead of index
            actual_label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            if actual_label == 1:
                mark = "Y"
            else:
                mark = "N"
            result_box.insert(tk.END, f"ID:{i:5d} - Fraud?: {mark} - Time: {end_time - start_time:.5f} sec\n")
            result_box.see(tk.END)
        
        y_pred_realtime.append(pred)
        progress_bar['value'] = i + 1
        root.update_idletasks()
    
    rt_total_time = time.perf_counter() - rt_process_start
    
    result_box.insert(tk.END, f"\n\n\n* REAL TIME PERFORMANCE *\n")
    result_box.insert(tk.END, f"Total samples processed: {len(y_pred_realtime)}\n")
    result_box.insert(tk.END, f"Total time: {rt_total_time:.2f} seconds\n")
    
    # Avoid division by zero if simulation canceled early
    if len(y_pred_realtime) > 0:
        result_box.insert(tk.END, f"Avg. time per prediction: {rt_total_time/len(y_pred_realtime):.5f} seconds\n")
        if rt_total_time > 0:
            result_box.insert(tk.END, f"Predictions per second: {len(y_pred_realtime)/rt_total_time:.1f}\n\n")
        else:
            result_box.insert(tk.END, f"Predictions per second: N/A (instantaneous)\n\n")
    else:
        result_box.insert(tk.END, "No predictions completed before cancellation.\n\n")
    root.update()
    
    if not cancel_event.is_set():
        metrics_process_start = time.perf_counter()
        root.update()
        
        y_pred_overall = model.predict(X_test_std)
        recall = recall_score(y_test, y_pred_overall)
        precision = precision_score(y_test, y_pred_overall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_total_time = time.perf_counter() - metrics_process_start

        # Calculate and display confusion matrix
        cm = confusion_matrix(y_test, y_pred_overall)
        
        # Extract confusion matrix values
        tn, fp, fn, tp = cm.ravel()
        
        # Create confusion matrix table
        result_box.insert(tk.END, f"* CONFUSION MATRIX *\n")
        result_box.insert(tk.END, f"True Positives (TP): {tp:5d} - Correctly identified fraud\n")
        result_box.insert(tk.END, f"False Positives (FP):{fp:5d} - Normal classified as fraud\n")
        result_box.insert(tk.END, f"False Negatives (FN):{fn:5d} - Fraud classified as normal\n")
        result_box.insert(tk.END, f"True Negatives (TN): {tn:5d} - Correctly identified normal\n\n")
        result_box.insert(tk.END, f"* OVERALL PERFORMANCE *\n")
        

        # Output results
        output = (
            f"Recall: {round(recall, 3)}\n"
            f"Precision: {round(precision, 3)}\n"
            f"F1 Score: {round(f1, 3)}\n"
        )
        result_box.insert(tk.END, output)
        result_box.see(tk.END)
    else:
        result_box.insert(tk.END, "\nSimulation was canceled - no metrics calculated.\n")
        result_box.see(tk.END)
        
    progress_bar.stop()
    progress_bar.config(mode='determinate')
    progress_bar['value'] = 0 

    # completion status
    if cancel_event.is_set():
        progress_label.config(text="Canceled")
    else:
        progress_label.config(text="Complete")
    
    # Re-enable buttons
    cancel_button.config(state=tk.DISABLED)
    if reset_button:
        reset_button.config(state=tk.NORMAL)

# UI control and cancellation
def cancel_simulation():
    global running_all_models, current_model_index
    cancel_event.set()
    lime_cancel_event.set()
    
    # If running all models stop the batch process
    if running_all_models:
        running_all_models = False
        current_model_index = 0
        result_box.insert(tk.END, "\nSimulation canceled by user.\n")
        result_box.see(tk.END)
        
        # Reset UI 
        progress_bar.stop()
        progress_bar.config(mode='determinate')
        progress_bar['value'] = 0
        progress_label.config(text="Canceled")
        
        # Re-enable all buttons
        run_all_models_button.config(state=tk.NORMAL)
        run_button.config(state=tk.NORMAL)
        cancel_button.config(state=tk.DISABLED)
        if reset_button:
            reset_button.config(state=tk.NORMAL)
        root.update()



# ============================================================================
# UI CONTROLS
# ============================================================================

# Train-test split config button
def open_settings_dialog():
    settings_window = tk.Toplevel(root)
    settings_window.title("Train-Test Split")
    settings_window.geometry("400x280")
    settings_window.resizable(False, False)
    settings_window.transient(root)
    settings_window.grab_set()
    
    # Center window
    settings_window.update_idletasks()
    width = settings_window.winfo_width()
    height = settings_window.winfo_height()
    x = (settings_window.winfo_screenwidth() // 2) - (width // 2)
    y = (settings_window.winfo_screenheight() // 2) - (height // 2)
    settings_window.geometry(f"{width}x{height}+{x}+{y}")

    # Create main frame 
    main_frame = tk.Frame(settings_window, padx=25, pady=25)
    main_frame.pack(fill='both', expand=True)
    
    # Train/test split percentage controls
    current_train_pct = train_test_split_ratio.get() * 100
    train_pct_var = tk.DoubleVar(value=current_train_pct)
    test_pct_label = tk.Label(main_frame, text=f"Test: {100-current_train_pct:.0f}%", font=font_l)
    
    def update_percentages(val):
        train_pct = float(val)
        test_pct = 100 - train_pct
        test_pct_label.config(text=f"Test: {test_pct:.0f}%")
    
    tk.Label(main_frame, text="Training Percentage:", font=font_l_b).pack(anchor='w', pady=(0, 8))
    train_slider = tk.Scale(main_frame, from_=50, to=90, orient='horizontal', variable=train_pct_var, command=update_percentages,font=font_m)
    train_slider.pack(fill='x', pady=(0, 8))
    test_pct_label.pack(anchor='w', pady=(0, 20))
    
    # Random seed controls
    tk.Label(main_frame, text="Random Seed:", font=font_l_b).pack(anchor='w', pady=(0, 8))
    seed_var = tk.IntVar(value=random_seed.get())
    seed_entry = tk.Entry(main_frame, textvariable=seed_var, width=15, font=font_m)
    seed_entry.pack(anchor='w', pady=(0, 25))
    
    # Apply app config settings
    def apply_settings():
    
        train_pct = train_pct_var.get()
        seed = seed_var.get()
        
        if 50 <= train_pct <= 90 and 1 <= seed <= 2147483647:
            train_test_split_ratio.set(train_pct / 100.0)
            random_seed.set(seed)
            result_box.insert(tk.END, f"Settings updated: {train_pct:.0f}% training, {100-train_pct:.0f}% testing, seed={seed}\n")
            settings_window.destroy()
        else:
            messagebox.showerror("Invalid Value", "Training: 50-90%, Seed: 1-2147483647")

    
    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(fill='x')
    
    tk.Button(button_frame, text="Cancel", command=settings_window.destroy, font=font_m).pack(side='right', padx=(5, 0))
    tk.Button(button_frame, text="Apply", command=apply_settings, font=font_m_b, bg='lightgreen').pack(side='right')

# Reset the application to initial state
def reset_application():
    global model_results, running_all_models, current_model_index, all_models_list
    global selected_model, selected_model_name, model_predictions
    
    # Cancel any ongoing operations
    cancel_event.set()
    lime_cancel_event.set()
    
    # Clear all results
    model_results = []
    model_predictions = {}
    
    # Disable results-dependent UI
    disable_results_dependent_ui()
    
    # Reset tracking variables
    running_all_models = False
    current_model_index = 0
    all_models_list = []
    
    # Reset UI elements
    model_selection.set("KNN")
    sampling_method.set("None")
    shap_model_selection.set("XGBoost")
    
    # Clear result box
    result_box.delete(1.0, tk.END)
    result_box.insert(tk.END, "Application reset. Ready to start fresh.\n")
    
    # Reset progress
    progress_label.config(text="Ready")
    progress_bar['value'] = 0
    
    # Reset cancellation events for fresh start
    cancel_event.clear()
    lime_cancel_event.clear()
    
    # Reset button states
    run_all_models_button.config(state=tk.NORMAL)
    cancel_button.config(state=tk.DISABLED)
    
    # Re-enable data-dependent UI
    if data_loaded:
        enable_data_dependent_ui()

# Get model from name
def get_model_from_selection(model_name):
    seed = random_seed.get()
    model_map = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=seed),
        "XGBoost": lambda: XGBClassifier(eval_metric='logloss'),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=30, random_state=seed),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=seed),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=3),
        "Neural Network": lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.1)
    }
    return model_map.get(model_name, lambda: KNeighborsClassifier(n_neighbors=3))

# Run selected model
def run_selected_model():
    # Reset cancel event
    cancel_event.clear()

    global X_train_std, X_test_std, y_train, y_test, selected_model_name

    # Get the model from the dropdown
    selected_model_name = model_selection.get()
    
    # Prepare data
    test_size = 1.0 - train_test_split_ratio.get()
    seed = random_seed.get()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Get model instance from selection
    model_factory = get_model_from_selection(selected_model_name)
    model_instance = model_factory()
    thread = threading.Thread(target=run_simulation_thread, args=(model_instance, selected_model_name))
    thread.daemon = True
    thread.start()

# Print results table in formatted display
def show_results_table():
    if not model_results:
        result_box.insert(tk.END, "\nNo batch processing results to display yet. Run batch processing first.\n\n")
        return

    # Set column width
    col_width = 16
    model_width = 20
    sampling_width = 18
    # Header
    header = (
        f"|{'Model':^{model_width}}"
        f"|{'Sampling':^{sampling_width}}"
        f"|{'Precision':^{col_width}}"
        f"|{'Recall':^{col_width}}"
        f"|{'F1':^{col_width}}"
        f"|{'Time (s)':^{col_width}}|\n"
    )
    separator = "|" + "|".join(["-" * model_width, "-" * sampling_width] + ["-" * col_width]*4) + "|\n"
    table = header + separator

    # Table rows
    for res in model_results:
        table += (
            f"|{res['Model']:<{model_width}}"
            f"|{res['Sampling']:<{sampling_width}}"
            f"|{round(res['Precision'],3):<{col_width}}"
            f"|{round(res['Recall'],3):<{col_width}}"
            f"|{round(res['F1'],3):<{col_width}}"
            f"|{res['Time (s)']:<{col_width}}|\n"
        )

    result_box.insert(tk.END, "\n\n * Model Results Table *\n")
    result_box.insert(tk.END, table)
    result_box.see(tk.END)



# ============================================================================
# BATCH PROCESSING
# ============================================================================

# Initialize and start batch processing
def run_all_models():
    global running_all_models, current_model_index, all_models_list
    
    # Reset cancel event
    cancel_event.clear()
    
    # Get random seed
    seed = random_seed.get()
    
    # Define models
    models_to_run = [

        # Logistic Regression with all sampling methods
        (lambda: LogisticRegression(max_iter=1000, random_state=seed), "Logistic Regression", "None"),
        (lambda: LogisticRegression(max_iter=1000, random_state=seed), "Logistic Regression", "SMOTE"),
        (lambda: LogisticRegression(max_iter=1000, random_state=seed), "Logistic Regression", "RandomOverSampler"),
        (lambda: LogisticRegression(max_iter=1000, random_state=seed), "Logistic Regression", "RandomUnderSampler"),
        
        # XGBoost with all sampling methods
        (lambda: XGBClassifier(eval_metric='logloss'), "XGBoost", "None"),
        (lambda: XGBClassifier(eval_metric='logloss'), "XGBoost", "SMOTE"),
        (lambda: XGBClassifier(eval_metric='logloss'), "XGBoost", "RandomOverSampler"),
        (lambda: XGBClassifier(eval_metric='logloss'), "XGBoost", "RandomUnderSampler"),
        
        # Random Forest with all sampling methods
        (lambda: RandomForestClassifier(n_estimators=30, random_state=seed), "Random Forest", "None"),
        (lambda: RandomForestClassifier(n_estimators=30, random_state=seed), "Random Forest", "SMOTE"),
        (lambda: RandomForestClassifier(n_estimators=30, random_state=seed), "Random Forest", "RandomOverSampler"),
        (lambda: RandomForestClassifier(n_estimators=30, random_state=seed), "Random Forest", "RandomUnderSampler"),
        
        # Decision Tree with all sampling methods
        (lambda: DecisionTreeClassifier(random_state=seed), "Decision Tree", "None"),
        (lambda: DecisionTreeClassifier(random_state=seed), "Decision Tree", "SMOTE"),
        (lambda: DecisionTreeClassifier(random_state=seed), "Decision Tree", "RandomOverSampler"),
        (lambda: DecisionTreeClassifier(random_state=seed), "Decision Tree", "RandomUnderSampler"),

        # KNN with all sampling methods
        (lambda: KNeighborsClassifier(n_neighbors=3), "KNN", "None"),
        (lambda: KNeighborsClassifier(n_neighbors=3), "KNN", "SMOTE"),
        (lambda: KNeighborsClassifier(n_neighbors=3), "KNN", "RandomOverSampler"),
        (lambda: KNeighborsClassifier(n_neighbors=3), "KNN", "RandomUnderSampler"),
        
        # Neural Network with all sampling methods
        (lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.1), "Neural Network", "None"),
        (lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.1), "Neural Network", "SMOTE"),
        (lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.1), "Neural Network", "RandomOverSampler"),
        (lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed, early_stopping=True, validation_fraction=0.1), "Neural Network", "RandomUnderSampler"),
    ]
    
    running_all_models = True
    current_model_index = 0
    all_models_list = models_to_run
    
    result_box.delete(1.0, tk.END)
    result_box.insert(tk.END, "Starting to run all 24 models (6 models × 4 sampling configurations)...\n\n")
    
    # Set progress bar for batch processing
    progress_bar.config(mode='determinate', maximum=24)
    progress_bar['value'] = 0
    progress_label.config(text="Starting batch processing...")
    
    # Clear previous results
    model_results.clear()
    
    # Disable results-dependent UI
    disable_results_dependent_ui()
    
    # Start running the first model
    run_next_model()

# Run the next model in batch processing
def run_next_model():
    global running_all_models, current_model_index, all_models_list
    
    if not running_all_models or cancel_event.is_set():
        if cancel_event.is_set():
            running_all_models = False
            current_model_index = 0
        return
    
    if current_model_index >= len(all_models_list):
        running_all_models = False
        result_box.insert(tk.END, "\n\n * COMPLETED * \n\n")
        result_box.insert(tk.END, f"Total models executed: {len(all_models_list)}\n")
        result_box.insert(tk.END, "You can now view the results table to compare all models.\n")
        result_box.see(tk.END)
        
        # Re-enable buttons
        run_all_models_button.config(state=tk.NORMAL)
        run_button.config(state=tk.NORMAL)
        cancel_button.config(state=tk.DISABLED)
        if fraud_analysis_button_ref:
            fraud_analysis_button_ref.config(state=tk.NORMAL)
        
        progress_label.config(text="All 24 models completed")
        progress_bar['value'] = 24
        return
    
    # Get current model info
    model_class, model_name, sampling_method_name = all_models_list[current_model_index]
    
    # Set sampling
    sampling_method.set(sampling_method_name)
    
    # Update selected model
    global selected_model, selected_model_name
    selected_model = model_class
    selected_model_name = model_name
    
    sampling_display = f" + {sampling_method_name}" if sampling_method_name != "None" else ""
    result_box.insert(tk.END, f"\n * Running Model {current_model_index + 1}/24: {model_name}{sampling_display} *\n")
    result_box.see(tk.END)
    
    # Prepare data for model
    global X_train_std, X_test_std, y_train, y_test
    test_size = 1.0 - train_test_split_ratio.get()
    seed = random_seed.get()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Apply sampling based on selected method
    if sampling_method_name == "SMOTE":
        sm = SMOTE(random_state=seed)
        X_train_std, y_train = sm.fit_resample(X_train_std, y_train)
        result_box.insert(tk.END, "SMOTE applied to training data.\n")
    elif sampling_method_name == "RandomOverSampler":
        ros = RandomOverSampler(random_state=seed)
        X_train_std, y_train = ros.fit_resample(X_train_std, y_train)
        result_box.insert(tk.END, "RandomOverSampler applied to training data.\n")
    elif sampling_method_name == "RandomUnderSampler":
        rus = RandomUnderSampler(random_state=seed)
        X_train_std, y_train = rus.fit_resample(X_train_std, y_train)
        result_box.insert(tk.END, "RandomUnderSampler applied to training data.\n")

    model_instance = model_class()
    
    # Start the simulation thread for model
    thread = threading.Thread(target=run_simulation_for_all_models, args=(model_instance, model_name, X_test_std.copy(), y_test.copy()))
    thread.daemon = True
    thread.start()

# Run simulation for a single model in batch processing
def run_simulation_for_all_models(model, model_name, X_test_data, y_test_data):
    global running_all_models, current_model_index
    
    if cancel_event.is_set():
        return
    
    # Disable buttons during simulation
    run_all_models_button.config(state=tk.DISABLED)
    run_button.config(state=tk.DISABLED)
    cancel_button.config(state=tk.NORMAL)
    if reset_button:
        reset_button.config(state=tk.DISABLED)
    if fraud_analysis_button_ref:
        fraud_analysis_button_ref.config(state=tk.DISABLED)
    
    progress_label.config(text=f"Running {model_name} ({current_model_index + 1}/24)...")
    progress_bar['value'] = current_model_index
    root.update()

    # Train model
    result_box.insert(tk.END, f"Training {model_name}...\n")
    root.update()
    
    if cancel_event.is_set():
        cleanup_on_cancel()
        return
    
    prediction_start = time.perf_counter()
    model.fit(X_train_std, y_train)
    
    if cancel_event.is_set():
        cleanup_on_cancel()
        return
    
    result_box.insert(tk.END, f"Running batch predictions for {len(X_test_data)} samples...\n")
    root.update()
    
    # Use batch prediction
    y_pred_batch = model.predict(X_test_data)
    fraud_count = sum(y_pred_batch)
    
    if cancel_event.is_set():
        cleanup_on_cancel()
        return
    
    result_box.insert(tk.END, f"Detected {fraud_count} fraudulent transactions\n")
    root.update()
    
    result_box.insert(tk.END, f"Calculating metrics for {model_name}...\n")
    root.update()
    
    # Use batch predictions for metrics
    recall = recall_score(y_test_data, y_pred_batch, zero_division=0)
    precision = precision_score(y_test_data, y_pred_batch, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Store predictions and probs for AUC ROC
    sampling_suffix = f" + {sampling_method.get()}" if sampling_method.get() != "None" else ""
    model_key = f"{model_name}{sampling_suffix}"
    
    # Get prediction probs for AUC calculation
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_proba = model.decision_function(X_test_data)
    else:
        # Fallback to binary predictions
        y_pred_proba = y_pred_batch
    
    model_predictions[model_key] = {
        'y_true': y_test_data,
        'y_pred_proba': y_pred_proba,
        'model_name': model_name,
        'sampling': sampling_method.get()
    }

    #time
    prediction_time = time.perf_counter() - prediction_start

    root.update()

    # Store results for table
    model_results.append({
        "Model": model_name,
        "Sampling": sampling_method.get(),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "Time (s)": round(prediction_time, 2)
    })

    # Enable results-dependent UI elements
    enable_results_dependent_ui()

    result_box.insert(tk.END, f"Completed - F1: {round(f1, 3)}, Time: {round(prediction_time, 2)}s\n")
    root.update()
    
    # Move to next model only if not cancelled
    if not cancel_event.is_set():
        current_model_index += 1
        progress_bar['value'] = current_model_index
        root.update()
        root.after(100, run_next_model)
    else:
        cleanup_on_cancel()

# UI cleanup and reset
def cleanup_on_cancel():
    global running_all_models, current_model_index
    
    running_all_models = False
    current_model_index = 0
    
    # Stop progress bar and reset UI
    progress_bar.stop()
    progress_bar.config(mode='determinate')
    progress_bar['value'] = 0
    progress_label.config(text="Canceled")
    
    # Re-enable all buttons
    run_all_models_button.config(state=tk.NORMAL)
    run_button.config(state=tk.NORMAL)
    cancel_button.config(state=tk.DISABLED)
    if reset_button:
        reset_button.config(state=tk.NORMAL)
    if fraud_analysis_button_ref:
        fraud_analysis_button_ref.config(state=tk.NORMAL)



# ============================================================================
# SHAP ANALYSIS
# ============================================================================

# SHAP visualization and plotting
def create_shap_plot(shap_values, X_sample, model_name):    
    # close matplotlib figures
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Handle different SHAP value formats
    if model_name in ["Random Forest", "Decision Tree"]:
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Format: [class_0_values, class_1_values]
            shap_values = shap_values[1]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
            # Format: (samples, features, classes) - take positive class
            shap_values = shap_values[:, :, 1]
    
    # Calculate mean absolute SHAP values for each feature
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    
    # Normalize SHAP values
    total_weight = np.sum(mean_shap_values)
    mean_shap_values = mean_shap_values / total_weight
    
    # Get feature names
    feature_names = X_sample.columns
    
    bars = ax.bar(feature_names, mean_shap_values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Normalized |SHAP Value|', fontsize=12)
    ax.set_title(f'SHAP Feature Importance - {model_name} (Normalized)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    for bar, val in zip(bars, mean_shap_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Run SHAP analysis for selected model
def run_shap_analysis():        
    # Start SHAP analysis in a separate thread
    shap_thread = threading.Thread(target=run_shap_analysis_thread)
    shap_thread.daemon = True
    shap_thread.start()

# SHAP analysis processing
def run_shap_analysis_thread():
    # Get selected model    
    selected_model_name = shap_model_selection.get()
    result_box.insert(tk.END, f"\nRunning SHAP Analysis for {selected_model_name}...\n")
    
    # Prepare data using configurable settings
    test_size = 1.0 - train_test_split_ratio.get()
    seed = random_seed.get()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the selected model
    if selected_model_name == "Random Forest" or selected_model_name == "Decision Tree":
        result_box.insert(tk.END, "This may take a few moments...\n")
    result_box.insert(tk.END, f"Training {selected_model_name} model...\n")
    root.update()
    model_factory = get_model_from_selection(selected_model_name)
    model = model_factory()
    model.fit(X_train_scaled, y_train)
    
    # Sample for SHAP (1000 samples for speed, as 2 are slow)
    X_sample = X.sample(n=1000, random_state=seed)
    X_sample_scaled = scaler.transform(X_sample)
    
    # Create SHAP explainer
    result_box.insert(tk.END, "Creating SHAP explainer...\n")
    root.update()
    
    if selected_model_name == "XGBoost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_scaled)
    elif selected_model_name in ["Random Forest", "Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_scaled)
    elif selected_model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer.shap_values(X_sample_scaled)
    
    # Create plot on main thread
    result_box.insert(tk.END, "Generating SHAP plot...\n")
    root.update()
    
    # Schedule plot creation
    root.after(0, lambda: create_shap_plot(shap_values, X_sample, selected_model_name))
    
    result_box.insert(tk.END, f"{selected_model_name} SHAP analysis completed\n\n")



# ============================================================================
# FRAUD ANALYSIS
# ============================================================================

# Run model and show fraud analysis window
def run_fraud_analysis():
    # Block LIME analysis for KNN
    if model_selection.get() == "KNN":
        messagebox.showerror("LIME Not Supported", "LIME analysis for KNN is not available due to inconsistent performance.")
        return
    
    # Open window, then start analysis
    fraud_window = create_fraud_window()
    
    # Start analysis thread
    fraud_thread = threading.Thread(target=run_fraud_analysis_thread, args=(fraud_window,))
    fraud_thread.daemon = True
    fraud_thread.start()

# Create fraud analysis window
def create_fraud_window():
    selected_model_name = model_selection.get()
    
    fraud_window = tk.Toplevel(root)
    fraud_window.title(f"Fraudulent Transactions - {selected_model_name}")
    fraud_window.geometry("1000x600")
    
    # track if window is closed
    fraud_window.is_closed = False
    fraud_window.cancel_event = threading.Event()
    
    # Handle window closing
    def on_closing():
        fraud_window.is_closed = True
        fraud_window.cancel_event.set()
        lime_cancel_event.set()
        fraud_window.destroy()
    
    fraud_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Status label
    status_label = tk.Label(fraud_window, text="Initializing fraud detection...", font=font_xl_b)
    status_label.pack(pady=10)
    
    # Instructions (initially hidden)
    instructions = tk.Label(fraud_window, text="Select a transaction to analyze with LIME:", font=font_xl_b)
    
    # Create frame for listbox and scrollbar
    list_frame = tk.Frame(fraud_window)
    list_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Scrollbar
    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side='right', fill='y')
    
    # Listbox for transactions
    transaction_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=20, font=font_m)
    transaction_listbox.pack(side='left', fill='both', expand=True)
    scrollbar.config(command=transaction_listbox.yview)
    
    # Button to analyze selected transaction
    def analyze_selected():
        selection = transaction_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a transaction to analyze.")
            return
            
        selected_idx = selection[0]
        selected_transaction = fraud_window.fraud_data.iloc[selected_idx].drop(['Actual_Class', 'Predicted_Class'])
        
        # Clear LIME cancellation event for new analysis
        lime_cancel_event.clear()
        
        # Start LIME analysis
        lime_thread = threading.Thread(target=run_lime_analysis_thread, args=(selected_transaction, fraud_window.model, fraud_window.scaler, fraud_window.model_name, selected_idx+1))
        lime_thread.daemon = True
        lime_thread.start()
    
    analyze_button = tk.Button(fraud_window, text="Analyze with LIME", 
                              command=analyze_selected, bg='lightgreen', 
                              font=font_m_b, state=tk.NORMAL)
    analyze_button.pack(pady=10)
    
    # Store references
    fraud_window.status_label = status_label
    fraud_window.instructions = instructions
    fraud_window.transaction_listbox = transaction_listbox
    fraud_window.analyze_button = analyze_button
    fraud_window.fraud_data = None
    fraud_window.model = None
    fraud_window.scaler = None
    fraud_window.model_name = model_selection.get()
    
    return fraud_window

# Run fraud detection analysis
def run_fraud_analysis_thread(fraud_window):

    # Get selected model and sampling method
    selected_model_name = model_selection.get()
    sampling_name = sampling_method.get()
    
    result_box.insert(tk.END, f"\nRunning Real-Time Fraud Analysis for {selected_model_name}...\n")
    
    # Disable fraud analysis button
    if fraud_analysis_button_ref:
        fraud_analysis_button_ref.config(state=tk.DISABLED)
    
    if fraud_window.is_closed or fraud_window.cancel_event.is_set():
        # Re-enable button if cancelled
        if fraud_analysis_button_ref:
            fraud_analysis_button_ref.config(state=tk.NORMAL)
        return
    
    # Update status safely
    def safe_update(update_func):
        if not fraud_window.is_closed:
            try:
                root.after(0, update_func)
            except tk.TclError:
                # Window was destroyed, mark as closed
                fraud_window.is_closed = True
    
    safe_update(lambda: fraud_window.status_label.config(text="Training model..."))
    
    # Prepare data
    test_size = 1.0 - train_test_split_ratio.get()
    seed = random_seed.get()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if fraud_window.cancel_event.is_set():
        return
    
    # Apply sampling if selected
    if sampling_name == "SMOTE":
        safe_update(lambda: fraud_window.status_label.config(text="Applying SMOTE sampling..."))
        smote = SMOTE(random_state=seed)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    elif sampling_name == "RandomOverSampler":
        safe_update(lambda: fraud_window.status_label.config(text="Applying Random oversampling..."))
        ros = RandomOverSampler(random_state=seed)
        X_train_scaled, y_train = ros.fit_resample(X_train_scaled, y_train)
    elif sampling_name == "RandomUnderSampler":
        safe_update(lambda: fraud_window.status_label.config(text="Applying Random undersampling..."))
        rus = RandomUnderSampler(random_state=seed)
        X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)
    
    if fraud_window.cancel_event.is_set():
        return
    
    # Train model
    safe_update(lambda: fraud_window.status_label.config(text=f"Training {selected_model_name} model..."))
    model_factory = get_model_from_selection(selected_model_name)
    model = model_factory()
    model.fit(X_train_scaled, y_train)
    
    if fraud_window.cancel_event.is_set():
        return
    
    # Store model and scaler in window for LIME analysis
    fraud_window.model = model
    fraud_window.scaler = copy.deepcopy(scaler)
    
    # Start real-time processing
    safe_update(lambda: fraud_window.status_label.config(text="Processing transactions in real-time..."))
    safe_update(lambda: fraud_window.instructions.pack(pady=10))
    
    # Initialize fraud data storage variable
    fraud_data_list = []
    fraud_count = 0
    
    # run dataset in real-time
    for idx, (test_idx, row) in enumerate(X_test.iterrows()):

        if fraud_window.cancel_event.is_set():
            break
            
        # Update status periodically on main thread
        if idx % 100 == 0:
            def update_status():
            
                if not fraud_window.is_closed:
                    fraud_window.status_label.config(text=f"Processing transaction {idx+1}/{len(X_test)}... Found {fraud_count} fraudulent")
  
            # Small delay to slow down very fast models
            time.sleep(0.1)
            safe_update(update_status)

        
        # Scale single transaction
        with scaler_lock:
            transaction_scaled = scaler.transform(pd.DataFrame([row]))
        
        # Make prediction for single transaction
        prediction = model.predict(transaction_scaled)[0]
            
        # Get prediction probability
        fraud_probability = model.predict_proba(transaction_scaled)[0][1]
            
        if prediction == 1:
            fraud_count += 1
            actual_class = y_test.loc[test_idx]
            transaction_data = row.copy()
            transaction_data['Actual_Class'] = actual_class
            transaction_data['Predicted_Class'] = prediction
            fraud_data_list.append(transaction_data)
            if actual_class == 1:
                mark = "Y"
            else:
                mark = "N"

            # Display in listbox with new transaction    
            display_text = f"ID:{idx:>6d} - Confidence:{fraud_probability:7.1%} - Fraud?: {mark}"
            safe_update(lambda text=display_text: fraud_window.transaction_listbox.insert(tk.END, text))
            fraud_window.fraud_data = pd.DataFrame(fraud_data_list)
    
    if fraud_window.cancel_event.is_set():
        # Re-enable button if cancelled
        if fraud_analysis_button_ref:
            fraud_analysis_button_ref.config(state=tk.NORMAL)
        return
    
    # Convert list to df for LIME analysis
    if fraud_data_list:
        fraud_window.fraud_data = pd.DataFrame(fraud_data_list)
        
        #  show complete
        safe_update(lambda: finalize_fraud_window(fraud_window, fraud_count))
        result_box.insert(tk.END, f"Real-time fraud analysis completed - Found {fraud_count} fraudulent transactions.\n")
    else:
        safe_update(lambda: fraud_window.status_label.config(text="No fraudulent transactions detected."))
        result_box.insert(tk.END, "No fraudulent transactions detected.\n")
    
    # Re-enable fraud analysis button after completed
    if fraud_analysis_button_ref:
        fraud_analysis_button_ref.config(state=tk.NORMAL)

# Update fraud window completion status
def finalize_fraud_window(fraud_window, count):
    # Check if window is still open
    if not fraud_window.is_closed:
        fraud_window.status_label.config(text=f"Loaded {count} fraudulent transactions. Select one to analyze.")



# ============================================================================
# LIME ANALYSIS
# ============================================================================

# Run LIME analysis for selected transaction
def run_lime_analysis_thread(transaction, model, scaler, model_name, transaction_num):

    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
        
    result_box.insert(tk.END, f"\nRunning LIME analysis for Transaction {transaction_num}...\n")
    root.update()
    
    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
    
    # Scale transaction
    with scaler_lock:
        transaction_scaled = scaler.transform(pd.DataFrame([transaction]))
    
    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
            
    # Use training data for explainer
    test_size = 1.0 - train_test_split_ratio.get()
    seed = random_seed.get()
    X_train, _, _, _ = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    
    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
    
    # Sample 10000 datapoints for faster LIME analysis (some models take a while, KNN is just hopeless)
    lime_sample_size = min(10000, len(X_train))
    X_train_sample = X_train.sample(n=lime_sample_size, random_state=seed)
    
    lime_scaler = StandardScaler()
    X_train_scaled = lime_scaler.fit_transform(X_train_sample)
    
    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
    
    explainer = LimeTabularExplainer(
        X_train_scaled,
        feature_names=X.columns,
        class_names=['Normal', 'Fraud'],
        mode='classification'
    )
    
    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
    
    # explanation
    explanation = explainer.explain_instance(
        transaction_scaled[0], 
        model.predict_proba,
        num_features=len(X.columns)
    )
    
    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
    
    # Extract feature importance and reorder by original feature order
    lime_data = explanation.as_list()
    
    # LIME feature list format: 'V1 <= -0.5', '-2 > V1 > -0.5', etc.
    # Map to original features by finding feature name
    importance_dict = {}
    for lime_feature, imp_val in lime_data:
        for dataset_feature in X.columns:
            # space is included afer {dataset_feature} in f string to avoid V10,V11, etc, matching V1
            if f"{dataset_feature} " in lime_feature:
                importance_dict[dataset_feature] = imp_val
                break
    
    # Reorder to match original feature order
    features = list(X.columns)
    importance = [importance_dict.get(feature, 0.0) for feature in features]
    
    if lime_cancel_event.is_set():
        result_box.insert(tk.END, f"LIME analysis for Transaction {transaction_num} was canceled.\n")
        return
    
    # Schedule LIME window on main thread
    root.after(0, lambda: show_lime_window(features, importance, model_name, transaction_num))

# Show LIME analysis results in new window
def show_lime_window(features, importance, model_name, transaction_num):

    if lime_cancel_event.is_set():
        return
    
    lime_window = tk.Toplevel(root)
    lime_window.title(f"LIME Analysis - {model_name} - Transaction {transaction_num}")
    lime_window.geometry("900x700")
    
    # Features are already in Time to Amount order, no reversal needed
    features_ordered = features
    importance_ordered = importance
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create vertical bar plot (flipped axes)
    colors = ['red' if imp < 0 else 'green' for imp in importance_ordered]
    bars = ax.bar(features_ordered, importance_ordered, color=colors)
    
    ax.set_ylabel('Feature Importance')
    ax.set_xlabel('Features')
    ax.set_title(f'LIME Feature Importance - {model_name}\nTransaction {transaction_num}')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    
    # Embed plot in tkinter window
    canvas = FigureCanvasTkAgg(fig, master=lime_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
    result_box.insert(tk.END, f"LIME analysis completed for Transaction {transaction_num}\n")



# ============================================================================
# USER INTERFACE
# ============================================================================

# Create UI
def create_ui():
    global cancel_button, run_all_models_button, run_button, show_table_button, reset_button, visualize_button

    # Data Loading
    data_frame = tk.Frame(root)
    data_frame.pack(fill='x', padx=10, pady=(5, 10))
    
    # Load CSV button
    load_csv_button = tk.Button(data_frame, text="Load Dataset", command=load_csv_file, bg='steelblue', fg='white', font=font_s_b, width=15)
    load_csv_button.pack(side='left', padx=(0, 10))
    
    # info and link
    info_frame = tk.Frame(data_frame)
    info_frame.pack(side='left', fill='x', expand=True)
    
    global status_label
    status_label = tk.Label(info_frame, text="No dataset loaded", font=font_s, fg='red')
    status_label.pack(anchor='w')
    
    link_label = tk.Label(info_frame, text="Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data", 
                         font=font_s, fg='blue', cursor='hand2')
    link_label.pack(anchor='w')
    
    # Make link clickable
    def open_kaggle_link(event):
        webbrowser.open("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data")
    link_label.bind("<Button-1>", open_kaggle_link)

    # Create main top frame for all control sections
    top_frame = tk.Frame(root)
    top_frame.pack(fill='x', padx=10, pady=10)

    # Model Config & Real Time section (combined top-left)
    model_realtime_frame = tk.LabelFrame(top_frame, text= "Real Time Simulation", padx=5, pady=5, font=font_m_b)
    model_realtime_frame.grid(row=0, column=0, padx=(0, 10), sticky='nsew')
    
    # grid layout
    # Row 0: Model and Run Selected Model
    tk.Label(model_realtime_frame, text="Model:", font=font_s).grid(row=0, column=0, sticky='w', padx=(20, 5))
    tk.Label(model_realtime_frame, text="Real Time Statistics:", font=font_s).grid(row=0, column=2, sticky='w', padx=(20, 20))
    run_button = tk.Button(model_realtime_frame, text="Run Model", command=run_selected_model, bg = 'cornflowerblue', font=font_s_b, width=24, state=tk.DISABLED if not data_loaded else tk.NORMAL)
    run_button.grid(row=1, column=2, padx=(20, 20), pady=(2, 10))
    
    # Row 1: Model dropdown
    model_dropdown = ttk.Combobox(model_realtime_frame, textvariable=model_selection, values=["Logistic Regression", "XGBoost", "Random Forest", "Decision Tree", "KNN", "Neural Network"],  state="readonly", width=14, font=font_s)
    model_dropdown.grid(row=1, column=0, sticky='w', padx=(20, 5), pady=(2, 10))
    model_dropdown.set("Logistic Regression")
    
    # Row 2: Sampling Method and Fraud Analysis
    tk.Label(model_realtime_frame, text="Sampling Method:", font=font_s).grid(row=2, column=0, sticky='w', padx=(20, 5))
    tk.Label(model_realtime_frame, text="Real Time with LIME:", font=font_s).grid(row=2, column=2, sticky='w', padx=(20, 20))
    fraud_analysis_button = tk.Button(model_realtime_frame, text="Notifications + LIME", command=run_fraud_analysis, bg='lightblue', font=font_s_b, width=24, state=tk.DISABLED if not data_loaded else tk.NORMAL)
    fraud_analysis_button.grid(row=3, column=2, padx=(20, 20), pady=(2, 0))
    
    # Row 3: Sampling dropdown
    sampling_dropdown = ttk.Combobox(model_realtime_frame, textvariable=sampling_method, values=["None", "SMOTE", "RandomOverSampler", "RandomUnderSampler"], state="readonly", width=14, font=font_s)
    sampling_dropdown.grid(row=3, column=0, sticky='w', padx=(20, 5), pady=(2, 0))
    sampling_dropdown.set("None")
    
    # Configure column weights
    model_realtime_frame.grid_columnconfigure(0, weight=0) 
    model_realtime_frame.grid_columnconfigure(1, weight=1)  
    model_realtime_frame.grid_columnconfigure(2, weight=0)  

    # Run model: Batch section
    batch_frame = tk.LabelFrame(top_frame, text="Batch Processing", padx=5, pady=5, font=font_m_b)
    batch_frame.grid(row=0, column=1, padx=10, sticky='nsew')
    
    run_all_models_button = tk.Button(batch_frame, text="Run Batch (24 total)", command=run_all_models, bg='lightseagreen', font=font_s_b, width=20, state=tk.DISABLED if not data_loaded else tk.NORMAL)
    run_all_models_button.pack(pady=(0, 5), fill='x')
    
    tk.Label(batch_frame, text="Batch Results:", font=font_s).pack(anchor='w', pady=(5, 2))
    show_table_button = tk.Button(batch_frame, text="Print Results Table", command=show_results_table, width=20, bg='lightgreen', font=font_s_b)
    show_table_button.pack(pady=(0, 5), fill='x')
    
    visualize_button = tk.Button(batch_frame, text="Graph Results", command=open_visualization_menu, width=20, bg='lightgreen', font=font_s_b)
    visualize_button.pack(fill='x')

    # Application Control section
    control_frame = tk.LabelFrame(top_frame, text="Application Control", padx=5, pady=5, font=font_m_b)
    control_frame.grid(row=0, column=2, padx=(10, 0), sticky='nsew')
    
    # Configuration button
    config_button = tk.Button(control_frame, text="Configure Train-Test Split", command=open_settings_dialog, bg = 'gold', font=font_s_b, width=20)
    config_button.pack(pady=(0, 5), fill='x')
    
    # Cancel button
    tk.Label(control_frame, text="Process Control:", font=font_s).pack(anchor='w', pady=(5, 2))
    cancel_button = tk.Button(control_frame, text="Cancel Current Simulation", command=cancel_simulation, bg = 'orange', state=tk.DISABLED, width=20, font=font_s_b)
    cancel_button.pack(pady=(0, 5), fill='x')
    
    reset_button = tk.Button(control_frame, text="Reset Application", command=reset_application, bg='red', font=font_s_b, width=20)
    reset_button.pack(fill='x')

    # Configure grid weights for responsive layout (3 columns)
    top_frame.grid_columnconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=1)
    top_frame.grid_columnconfigure(2, weight=1)

    # SHAP Analysis
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(fill='x', padx=10, pady=(0, 10))
    
    shap_frame = tk.LabelFrame(bottom_frame, text="SHAP Analysis", padx=5, pady=5, font=font_m_b)
    shap_frame.pack(side='left', anchor='w')
    
    shap_model_frame = tk.Frame(shap_frame)
    shap_model_frame.pack(fill='x', pady=(0, 5))
    
    tk.Label(shap_model_frame, text="Model:", font=font_s).pack(side='left', padx=(0, 5))
    shap_model_dropdown = ttk.Combobox(shap_model_frame, textvariable=shap_model_selection, values=["Logistic Regression", "XGBoost", "Random Forest", "Decision Tree"],state="readonly", width=18, font=font_s)
    shap_model_dropdown.pack(side='left')
    shap_model_dropdown.set("Logistic Regression")
    
    shap_button = tk.Button(shap_frame, text="Run SHAP Analysis", command=run_shap_analysis, bg='plum', width=20, font=font_s_b,state=tk.DISABLED if not data_loaded else tk.NORMAL)
    shap_button.pack(fill='x')
    
    # Store global references for enabling/disabling
    global shap_button_ref, fraud_analysis_button_ref
    shap_button_ref = shap_button
    fraud_analysis_button_ref = fraud_analysis_button

    # Progress bar section (centered inline with SHAP)
    progress_frame = tk.Frame(bottom_frame)
    progress_frame.pack(side='right', expand=True)
    
    # Center the progress components within the progress_frame
    progress_center_frame = tk.Frame(progress_frame)
    progress_center_frame.pack(expand=True)
    
    # Progress widgets
    global progress_label, progress_bar
    progress_label = tk.Label(progress_center_frame, text="Ready", font=font_m_b)
    progress_label.pack(anchor='center')
    
    progress_bar = ttk.Progressbar(progress_center_frame, orient='horizontal', mode='determinate', length=400)
    progress_bar.pack(anchor='center', pady=(5, 0))

    # Text/output box
    global result_box
    result_box = scrolledtext.ScrolledText(root, width=160, height=30, font=font_code)
    result_box.pack(pady=(0, 10), padx=10, fill='both', expand=True)



# ============================================================================
# RESULTS DISPLAY
# ============================================================================

# Get results of dataframe
def get_results_df():
    if not model_results:
        return None
    df = pd.DataFrame(model_results)
    return df

# Show message if no results
def _show_df_empty_msg():
    messagebox.showinfo("No Batch Results", "No batch processing results available yet. Run batch processing first.")



# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Choose a plot to display
def open_visualization_menu():
    df = get_results_df()
    if df is None:
        _show_df_empty_msg()
        return
    
    menu = tk.Toplevel(root)
    menu.title("Visualize Results")
    menu.geometry("380x220")

    tk.Label(menu, text="Choose a plot to display:", font=font_l_b).pack(pady=8)

    btn_frame = tk.Frame(menu)
    btn_frame.pack(pady=4, fill='x', padx=10)

    tk.Button(btn_frame, text="Bar: Metric by Model", width=30, font=font_s_b, command=lambda: plot_metric_by_model('F1')).pack(pady=4)
    tk.Button(btn_frame, text="Bar: Precision/Recall/F1", width=30, font=font_s_b, command=lambda: plot_metrics_grouped()).pack(pady=4)
    tk.Button(btn_frame, text="Batch Processing Times", width=30, font=font_s_b, command=plot_timings).pack(pady=4)
    tk.Button(btn_frame, text="AUC ROC Curves", width=30, font=font_s_b, command=plot_auc_roc_curves).pack(pady=4)

# Window for general plot display
def _create_plot_window(title, fig):
    win = tk.Toplevel(root)
    win.title(title)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill='both', expand=True)
    win.geometry("1000x700")
    return win

# F1 scores for all models
def plot_metric_by_model(metric='F1'):
    df = get_results_df()
    if df is None:
        _show_df_empty_msg()
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9,6))
    # Sampling column always exists
    bars = sns.barplot(data=df, x='Model', y=metric, hue='Sampling', errorbar=None, ax=ax)
    ax.legend(loc='best')
    ax.set_title(f"{metric} by Model (grouped by Sampling)")
    ax.set_ylabel(metric)
    ax.set_ylim(0,1 if metric in ['Precision','Recall','F1'] else None)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=2, fontsize=8)
    _create_plot_window(f"{metric} by Model", fig)

# Metrics for models (no sampling)
def plot_metrics_grouped():
    df = get_results_df()
    if df is None:
        _show_df_empty_msg()
        return
    # Filter to only show no sampling
    df_filtered = df[df['Sampling'] == 'None'].copy()

    metrics = [c for c in ['Precision','Recall','F1'] if c in df_filtered.columns]
    
    # convert df to long format, each value to be plotted is its own row
    long = df_filtered.melt(id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Value')
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = sns.barplot(data=long, x='Model', y='Value', hue='Metric', errorbar=None, ax=ax)
    ax.set_title("Precision / Recall / F1 by Model (No Sampling)")
    ax.set_ylim(0,1)
    ax.legend(loc='best')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=2, fontsize=8)
    _create_plot_window("Precision/Recall/F1 by Model (No Sampling)", fig)

# AUC ROC curves
def plot_auc_roc_curves():

    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown']
    model_names = ['KNN', 'XGBoost', 'Random Forest', 'Logistic Regression', 'Decision Tree', 'Neural Network']
    
    # Track which lines are visible
    lines = {}
    legend_labels = {}
    
    # Plot ROC curve for each model
    color_idx = 0
    for model_name in model_names:
        model_key = model_name
        
        if model_key in model_predictions:
            data = model_predictions[model_key]
            y_true = data['y_true']
            y_pred_proba = data['y_pred_proba']
            
            # Calculate curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot curve
            color = colors[color_idx % len(colors)]
            line, = ax.plot(fpr, tpr, color=color, linewidth=2, 
                          label=f'{model_name} (AUC = {roc_auc:.3f})')
            lines[model_name] = line
            legend_labels[model_name] = f'{model_name} (AUC = {roc_auc:.3f})'
            color_idx += 1
    
    # random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Different Models (No Sampling)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create interactive legend
    legend = ax.legend(loc='lower right', fontsize=10)
    legend.set_draggable(True)
    
    # Interactive legend fn
    def on_legend_click(event):

        if legend.get_window_extent().contains(event.x, event.y):
            for i, legend_line in enumerate(legend.get_lines()):
                if legend_line.contains(event)[0]:
                    legend_text = legend.get_texts()[i].get_text()
                    model_name = legend_text.split(' (AUC')[0]
                    if model_name in lines:
                        line = lines[model_name]
                        line.set_visible(not line.get_visible())
                        legend_text_obj = legend.get_texts()[i]
                        if line.get_visible():
                            legend_text_obj.set_alpha(1.0)
                            legend_line.set_alpha(1.0)
                        else:
                            legend_text_obj.set_alpha(0.3)
                            legend_line.set_alpha(0.3)
                        fig.canvas.draw()
                    break
    fig.canvas.mpl_connect('button_press_event', on_legend_click)
    
    ax.text(0.02, 0.98, 'Click legend items to hide/show curves', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    _create_plot_window("AUC ROC Curves", fig)

# Batch processing times
def plot_timings():
    df = get_results_df()
    if df is None:
        _show_df_empty_msg()
        return
    time_cols = [c for c in ['Time (s)'] if c in df.columns]
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10,6))
    bars = sns.barplot(data=df, x='Model', y='Time (s)', hue='Sampling', errorbar=None, ax=ax)
    ax.legend(loc='best')
    ax.set_title("Batch Processing Time by Model")
    ax.set_ylabel('Seconds')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=2, fontsize=8)
    _create_plot_window("Batch Timings by Model", fig)


if __name__ == "__main__":
    create_ui()
    # brute-force shutdown (some processes can hang indefinitely even on exit)
    def on_closing():
        os._exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Initialize with data-dependent UI disabled
    disable_data_dependent_ui()
    root.mainloop()
