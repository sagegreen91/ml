
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
file = input("Enter dataset filename: ")
data = pd.read_csv(file, encoding="latin1")

# Remove missing values
data = data.dropna(subset=["CustomerID"])

# Create customer features
data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
customer = data.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalPrice": "sum",
    "InvoiceNo": "nunique"
}).reset_index()

customer.columns = ["CustomerID", "Quantity", "TotalPrice", "Purchases"]

# Create customer segments based on spending
customer["Segment"] = (
    customer["TotalPrice"] >= customer["TotalPrice"].median()
).astype(int)

X = customer[["Quantity", "TotalPrice", "Purchases"]]
y = customer["Segment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train decision tree using entropy
model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Visualize tree
plt.figure(figsize=(14, 7))
plot_tree(model, feature_names=X.columns,
          class_names=["Low Spending", "High Spending"],
          filled=True, rounded=True)
plt.show()

# Feature importance
print("Feature Importance:")
for name, value in zip(X.columns, model.feature_importances_):
    print(name, ":", value)
