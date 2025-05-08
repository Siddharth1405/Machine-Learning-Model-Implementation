import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("sms.tsv", sep='\t', header=None, names=['label', 'message'])

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

predictions = model.predict(X_test_counts)

print(f"\nModel Accuracy: {accuracy_score(y_test, predictions)*100:.1f}%\n")

test_messages = [
    "Free prize! Click now", 
    "Hi Bob, meeting tomorrow",
    "You won $1000! Claim your prize",
    "Reminder: Team lunch at 1pm"
]

print("--- Message Classification Results ---")
for msg in test_messages:
    prediction = model.predict(vectorizer.transform([msg]))[0]
    result = "SPAM MESSAGE!" if prediction == 1 else "Normal message"
    print(f"\nMessage: '{msg}'\nClassification: {result}")
