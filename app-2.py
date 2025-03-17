import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Initialize main window
root = tk.Tk()
root.title("Detection of Autism Spectrum Disorder Using Machine Learning Techniques")
root.geometry("1000x700")
root.config(bg="white")

# Define fonts
font = ("Times", 16, "bold")
font1 = ("Times", 13, "bold")

# Global variables
dataset = None
test_dataset = None
X_train, X_test, y_train, y_test = None, None, None, None
scaler = StandardScaler()

# Title label
title = tk.Label(
    root,
    text="Detection of Autism Spectrum Disorder Using Machine Learning Techniques",
    justify=tk.LEFT,
    bg="lavender blush",
    fg="black",
    font=font,
    height=3,
    width=100
)
title.pack(pady=10)

# Frame for buttons
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=10)

# Button style dictionary
button_style = {"font": font1, "width": 25, "height": 2, "fg": "white"}

# Functions
def upload_dataset():
    global dataset
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            dataset = pd.read_csv(file_path)
            output_text.insert(tk.END, "Dataset uploaded successfully!\n")
            # Debugging: Print dataset shape and class distribution
            output_text.insert(tk.END, f"Dataset shape: {dataset.shape}\n")
            output_text.insert(tk.END, f"Class distribution:\n{dataset.iloc[:, -1].value_counts()}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

def upload_test_data():
    global test_dataset
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            test_dataset = pd.read_csv(file_path)
            output_text.insert(tk.END, "Test dataset uploaded successfully!\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load test dataset: {e}")

def preprocess_data():
    global dataset, X_train, X_test, y_train, y_test, scaler
    if dataset is not None:
        try:
            # Handle missing data
            dataset.replace("?", pd.NA, inplace=True)
            dataset.dropna(inplace=True)

            # Encode categorical data
            for column in dataset.columns:
                if dataset[column].dtype == 'object':
                    dataset[column] = dataset[column].astype('category').cat.codes

            # Split data into features and target
            X = dataset.iloc[:, :-1].values  # Features (all columns except the last)
            y = dataset.iloc[:, -1].values   # Target (last column)

            # Debugging: Print feature and target shapes
            output_text.insert(tk.END, f"Features shape: {X.shape}\n")
            output_text.insert(tk.END, f"Target shape: {y.shape}\n")

            # Scale features
            X = scaler.fit_transform(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            output_text.insert(tk.END, "Data preprocessed successfully!\n")
            # Debugging: Print train-test split sizes
            output_text.insert(tk.END, f"Training set size: {X_train.shape}\n")
            output_text.insert(tk.END, f"Testing set size: {X_test.shape}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess data: {e}")
    else:
        output_text.insert(tk.END, "Error: Please upload a dataset first!\n")

def run_logistic_regression():
    if X_train is not None:
        try:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            display_metrics("Logistic Regression", y_test, y_pred)
            # Debugging: Evaluate on training data
            y_train_pred = model.predict(X_train)
            output_text.insert(tk.END, f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.2f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train Logistic Regression: {e}")
    else:
        output_text.insert(tk.END, "Error: Please preprocess the data first!\n")

def run_svm():
    if X_train is not None:
        try:
            model = SVC()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            display_metrics("SVM", y_test, y_pred)
            # Debugging: Evaluate on training data
            y_train_pred = model.predict(X_train)
            output_text.insert(tk.END, f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.2f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train SVM: {e}")
    else:
        output_text.insert(tk.END, "Error: Please preprocess the data first!\n")

def run_ann():
    if X_train is not None:
        try:
            model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            display_metrics("ANN", y_test, y_pred)
            # Debugging: Evaluate on training data
            y_train_pred = model.predict(X_train)
            output_text.insert(tk.END, f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.2f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train ANN: {e}")
    else:
        output_text.insert(tk.END, "Error: Please preprocess the data first!\n")

def detect_autism():
    global test_dataset, X_train, y_train, scaler
    if test_dataset is not None and X_train is not None:
        try:
            # Preprocess test data
            test_dataset.replace("?", pd.NA, inplace=True)
            test_dataset.dropna(inplace=True)
            for column in test_dataset.columns:
                if test_dataset[column].dtype == 'object':
                    test_dataset[column] = test_dataset[column].astype('category').cat.codes

            # Ensure test data has the same features as training data
            train_columns = dataset.columns[:-1]  # Exclude target column
            for col in train_columns:
                if col not in test_dataset.columns:
                    test_dataset[col] = 0  # Fill missing columns with default values
            test_dataset = test_dataset[train_columns]  # Reorder columns to match training data

            # Scale test data
            test_features = scaler.transform(test_dataset.values)

            # Train a Logistic Regression model (or any other model)
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(test_features)
            prediction_labels = ["Autistic" if pred == 1 else "Non-Autistic" for pred in predictions]

            # Display predictions with case numbers
            output_text.insert(tk.END, "Autism Predictions:\n")
            for case_no, prediction in enumerate(prediction_labels, start=1):
                output_text.insert(tk.END, f"Case {case_no}: {prediction}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect autism: {e}")
    else:
        output_text.insert(tk.END, "Error: Please upload a test dataset and preprocess the data first!\n")

def display_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    fscore = f1_score(y_true, y_pred, average='weighted')

    metrics_text = f"{model_name} Metrics:\n"
    metrics_text += f"Accuracy: {accuracy:.2f}\n"
    metrics_text += f"Precision: {precision:.2f}\n"
    metrics_text += f"Recall: {recall:.2f}\n"
    metrics_text += f"F-Score: {fscore:.2f}\n"
    metrics_text += "--------------------------\n"

    output_text.insert(tk.END, metrics_text)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, model_name)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

# Buttons
upload_button = tk.Button(button_frame, text="Upload ASD Dataset", command=upload_dataset, bg="#4CAF50", **button_style)
upload_button.grid(row=0, column=0, padx=10, pady=10)

preprocess_button = tk.Button(button_frame, text="Preprocess Data", command=preprocess_data, bg="#FF9800", **button_style)
preprocess_button.grid(row=0, column=1, padx=10, pady=10)

lr_button = tk.Button(button_frame, text="Run Logistic Regression", command=run_logistic_regression, bg="#2196F3", **button_style)
lr_button.grid(row=0, column=2, padx=10, pady=10)

svm_button = tk.Button(button_frame, text="Run SVM Algorithm", command=run_svm, bg="#9C27B0", **button_style)
svm_button.grid(row=1, column=0, padx=10, pady=10)

ann_button = tk.Button(button_frame, text="Run ANN Algorithm", command=run_ann, bg="#E91E63", **button_style)
ann_button.grid(row=1, column=1, padx=10, pady=10)

upload_test_button = tk.Button(button_frame, text="Upload Test Data", command=upload_test_data, bg="#795548", **button_style)
upload_test_button.grid(row=2, column=0, padx=10, pady=10)

detect_button = tk.Button(button_frame, text="Detect Autism", command=detect_autism, bg="#673AB7", **button_style)
detect_button.grid(row=2, column=1, padx=10, pady=10)

exit_button = tk.Button(button_frame, text="Exit", command=root.quit, bg="#F44336", **button_style)
exit_button.grid(row=2, column=2, padx=10, pady=10)

# Output text box with a scrollbar
output_frame = tk.Frame(root)
output_frame.pack(pady=10)

output_text = tk.Text(output_frame, height=20, width=100, font=("Times", 12, "bold"))
output_text.pack()

scroll = tk.Scrollbar(output_text)
output_text.configure(yscrollcommand=scroll.set)

# Run the application
root.mainloop()