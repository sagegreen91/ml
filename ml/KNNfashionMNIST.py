
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset from pendrive
file = input("Enter dataset filename: ")
data = pd.read_csv(file)

# First column is label, remaining columns are pixels
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# Normalize pixel values
X = X / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Test different K values
for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nK =", k)
    print("Accuracy:", accuracy_score(y_test, pred))
