from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

"""# Load local dataset folder
folder = "path"
data = load_files(folder, encoding="utf-8", decode_error="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)"""
# Convert text into word counts
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

y_train = train.target
y_test = test.target

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred1 = mnb.predict(X_test)

print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1-score:", f1_score(y_test, pred1, average="weighted"))

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
pred2 = bnb.predict(X_test)

print("\nBernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1-score:", f1_score(y_test, pred2, average="weighted"))
