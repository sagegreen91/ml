
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Load dataset from pendrive
file = input("Enter dataset filename: ")
data = pd.read_csv(file, header=None)

# First column is label, remaining columns are pixels
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# Use a smaller sample to reduce running time
data = data.sample(n=min(3000, len(data)), random_state=42)
X = data.iloc[:, 1:] / 255.0
y = data.iloc[:, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compare SVM kernels
for kernel in ["linear", "poly", "rbf"]:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nKernel:", kernel)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1-score:", f1_score(y_test, pred, average="weighted"))
