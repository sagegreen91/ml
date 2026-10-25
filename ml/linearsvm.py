
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset from pendrive
file = input("Enter dataset filename: ")
data = pd.read_csv(file)

# Use two features for 2D visualization
X = data.iloc[:, [0, 1]]
y = (data.iloc[:, -1] != "Iris-setosa").astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear SVM
model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Create decision boundary
x_min, x_max = X_train[:, 0].min()-1, X_train[:, 0].max()+1
y_min, y_max = X_train[:, 1].min()-1, X_train[:, 1].max()+1
xx, yy = __import__("numpy").meshgrid(
    __import__("numpy").linspace(x_min, x_max, 200),
    __import__("numpy").linspace(y_min, y_max, 200)
)
Z = model.predict(__import__("numpy").c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.contour(
    xx, yy,
    model.decision_function(__import__("numpy").c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
    levels=[-1, 0, 1], linestyles=["--", "-", "--"]
)
plt.title("Linear SVM: Setosa vs Non-Setosa")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.show()
