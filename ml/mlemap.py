
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
file = input("Enter dataset filename: ")

if file.endswith(".csv"):
    data = pd.read_csv(file)
    texts = data.iloc[:, 0].astype(str).tolist()
else:
    with open(file, "r", encoding="utf-8") as f:
        texts = [f.read()]

# Convert text into word counts
vectorizer = CountVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(texts)

# Word counts
counts = np.asarray(X.sum(axis=0)).flatten()
V = len(counts)
N = counts.sum()

# MLE
mle = counts / N
print("First 5 MLE probabilities:", mle[:5])

# MAP with different priors
for alpha in [0.1, 1, 10]:
    map_est = (counts + alpha) / (N + alpha * V)
    print("\nAlpha:", alpha)
    print("First 5 MAP probabilities:", map_est[:5])
