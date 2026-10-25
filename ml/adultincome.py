
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset from pendrive
file = input("Enter dataset filename: ")

cols = ["Age", "Workclass", "Fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Sex",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry",
        "Income"]

data = pd.read_csv(file, names=cols, skipinitialspace=True, na_values="?")

# Preprocess data
data = data.dropna()
X = data.drop("Income", axis=1)
y = data["Income"].str.strip()

# Convert categorical columns into numbers
X = pd.get_dummies(X)
y = y.map({"<=50K": 0, ">50K": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
pred1 = lr.predict(X_test_scaled)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
pred2 = dt.predict(X_test)

# Evaluate both models
for name, pred in [("Logistic Regression", pred1),
                   ("Decision Tree", pred2)]:
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1-score:", f1_score(y_test, pred, zero_division=0))

# Feature importance for Decision Tree
print("\nTop Decision Tree features:")
importance = pd.Series(dt.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(10))
