from tkinter import messagebox, filedialog, ttk
from tkinter import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# Global Variables
dataset = None
test_data = None
model = None
le = LabelEncoder()
X_train, X_test, y_train, y_test = None, None, None, None
rmse_values = {}

# Create main window
root = Tk()
root.title("Crop Yield Prediction")
root.geometry("1200x700")
root.configure(bg="lightgrey")

# Function to open new pages
def open_page(page_name):
    for widget in root.winfo_children():
        widget.destroy()
    page_name()

# Home Page
def home_page():
    Label(root, text="Crop Yield Prediction & Fertilizer Suggestion", bg='black', fg='white', font=("Arial", 20, "bold")).pack(fill=X, pady=10)
    
    Button(root, text="📂 Upload Dataset", command=lambda: open_page(upload_page), font=("Arial", 14), bg="blue", fg="white", width=30).pack(pady=10)
    Button(root, text="🔄 Preprocess Dataset", command=lambda: open_page(preprocess_page), font=("Arial", 14), bg="green", fg="white", width=30).pack(pady=10)
    Button(root, text="🤖 Train Model", command=lambda: open_page(train_page), font=("Arial", 14), bg="orange", fg="white", width=30).pack(pady=10)
    Button(root, text="📊 Upload Test Data & Predict", command=lambda: open_page(predict_page), font=("Arial", 14), bg="purple", fg="white", width=30).pack(pady=10)
    Button(root, text="📈 Show RMSE Graph", command=graph, font=("Arial", 14), bg="red", fg="white", width=30).pack(pady=10)
    Button(root, text="❌ Exit", command=root.quit, font=("Arial", 14), bg="black", fg="white", width=30).pack(pady=10)

# Upload Dataset Page
def upload_page():
    global dataset

    def upload():
        global dataset
        filename = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV files", "*.csv")])
        if not filename:
            messagebox.showerror("Error", "No file selected!")
            return

        dataset = pd.read_csv(filename)
        dataset.fillna(0, inplace=True)
        
        Label(root, text="✅ Dataset Uploaded Successfully!", font=("Arial", 12), fg="green").pack()

        # Show dataset
        show_table(dataset)

    Label(root, text="Upload Dataset", font=("Arial", 20, "bold")).pack(pady=10)
    Button(root, text="📂 Choose File", command=upload, font=("Arial", 14), bg="blue", fg="white").pack(pady=10)
    Button(root, text="⬅ Back", command=lambda: open_page(home_page), font=("Arial", 14), bg="black", fg="white").pack(pady=10)

# Show dataset in table
def show_table(data):
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=1)

    tree = ttk.Treeview(frame, columns=list(data.columns), show="headings")
    
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    for i, row in data.iterrows():  # Show full dataset
        tree.insert("", "end", values=list(row))

    tree.pack(fill=BOTH, expand=1)

# Preprocess Dataset Page
def preprocess_page():
    def processDataset():
        global dataset, X_train, X_test, y_train, y_test
        if dataset is None:
            messagebox.showerror("Error", "Upload dataset first!")
            return

        dataset['State_Name'] = le.fit_transform(dataset['State_Name'])
        dataset['District_Name'] = le.fit_transform(dataset['District_Name'])
        dataset['Season'] = le.fit_transform(dataset['Season'])
        dataset['Crop'] = le.fit_transform(dataset['Crop'])

        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.astype('uint8')

        X = normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        Label(root, text="✅ Dataset Preprocessed Successfully!", font=("Arial", 12), fg="green").pack()
        show_table(dataset)

    Label(root, text="Preprocess Dataset", font=("Arial", 20, "bold")).pack(pady=10)
    Button(root, text="🔄 Start Preprocessing", command=processDataset, font=("Arial", 14), bg="green", fg="white").pack(pady=10)
    Button(root, text="⬅ Back", command=lambda: open_page(home_page), font=("Arial", 14), bg="black", fg="white").pack(pady=10)

# Train Model Page
def train_page():
    def trainModel():
        global model, rmse_values
        if X_train is None or X_test is None:
            messagebox.showerror("Error", "Preprocess dataset first!")
            return

        rmse_values = {}

        models = {
            "Decision Tree": DecisionTreeRegressor(max_depth=100, random_state=0, max_leaf_nodes=20, max_features=5, splitter="random"),
            "Linear Regression": LinearRegression(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
        }

        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            predict = mdl.predict(X_test)

            mse = np.mean((predict - y_test) ** 2)
            rmse = np.sqrt(mse) / 1000

            rmse_values[name] = rmse

        model = models["Decision Tree"]
        
        Label(root, text="✅ Model Trained Successfully! Training Data:", font=("Arial", 12), fg="green").pack()
        show_table(pd.DataFrame(X_train))

    Label(root, text="Train Model", font=("Arial", 20, "bold")).pack(pady=10)
    Button(root, text="🤖 Train", command=trainModel, font=("Arial", 14), bg="orange", fg="white").pack(pady=10)
    Button(root, text="⬅ Back", command=lambda: open_page(home_page), font=("Arial", 14), bg="black", fg="white").pack(pady=10)

# Graph function
def graph():
    if not rmse_values:
        messagebox.showerror("Error", "Train models first!")
        return
    
    bars = list(rmse_values.keys())
    height = list(rmse_values.values())

    plt.figure(figsize=(8, 5))
    plt.bar(bars, height, color=['blue', 'green', 'red'])
    plt.xlabel("ML Models")
    plt.ylabel("RMSE Value")
    plt.title("Model RMSE Comparison")
    plt.show()

# Test Data Upload and Prediction
def predict_page():
    def upload_test_data():
        global test_data
        filename = filedialog.askopenfilename(title="Select Test Data", filetypes=[("CSV files", "*.csv")])
        if not filename:
            messagebox.showerror("Error", "No file selected!")
            return

        test_data = pd.read_csv(filename)
        test_data.fillna(0, inplace=True)
        Label(root, text="✅ Test Data Uploaded Successfully!", font=("Arial", 12), fg="green").pack()
        show_table(test_data)

    Label(root, text="Upload Test Data", font=("Arial", 20, "bold")).pack(pady=10)
    Button(root, text="📂 Choose File", command=upload_test_data, font=("Arial", 14), bg="purple", fg="white").pack(pady=10)
    Button(root, text="⬅ Back", command=lambda: open_page(home_page), font=("Arial", 14), bg="black", fg="white").pack(pady=10)

# Initialize Home Page
home_page()
root.mainloop()