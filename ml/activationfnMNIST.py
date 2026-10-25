
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset from pendrive
file = input("Enter dataset filename: ")
data = pd.read_csv(file, header=None)

# First column is label, remaining columns are pixels
y = data.iloc[:, 0]
X = data.iloc[:, 1:] / 255.0

# Use a smaller sample for faster execution
data = data.sample(n=min(5000, len(data)), random_state=42)
y = data.iloc[:, 0]
X = data.iloc[:, 1:] / 255.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compare activation functions
for act in ["logistic", "relu", "tanh"]:
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation=act,
        max_iter=30,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    pred = model.predict(X_test)

    print("\nActivation:", act)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Training time:", round(elapsed, 2), "seconds")
    print("Iterations:", model.n_iter_)
