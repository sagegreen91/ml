
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset from pendrive
file = input("Enter dataset filename: ")
data = pd.read_csv(file, sep=";")

# Separate features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Test different MLP architectures
for layers in [(5,), (10,), (10, 5), (20, 10)]:
    model = MLPClassifier(
        hidden_layer_sizes=layers, max_iter=500, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nHidden layers:", layers)
    print("Accuracy:", accuracy_score(y_test, pred))
