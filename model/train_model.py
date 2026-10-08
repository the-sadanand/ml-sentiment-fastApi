"""Train a small TF-IDF + Logistic Regression sentiment model and serialize it.

Usage:
    python model/train_model.py

The artifact is written to MODEL_PATH (default: ./model/sentiment_model.pkl).
A built-in toy dataset is used so training is fully reproducible offline.
"""
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

POSITIVE = [
    "I love this platform!", "This is absolutely fantastic", "Great product, works perfectly",
    "I am so happy with the service", "Wonderful experience, highly recommend",
    "Amazing quality and fast delivery", "The support team was incredibly helpful",
    "Best purchase I have made this year", "Excellent value for the money",
    "I really enjoyed using this app", "Brilliant design and very intuitive",
    "Totally satisfied, five stars", "What a delightful surprise, loved it",
    "Super smooth and reliable", "This exceeded all my expectations",
    "Fantastic movie with a beautiful story", "The food was delicious and the staff friendly",
    "I'm thrilled with the results", "Outstanding performance, great job",
    "Very pleased, will buy again",
]
NEGATIVE = [
    "I hate this platform", "This is absolutely terrible", "Awful product, broke after one day",
    "I am very disappointed with the service", "Horrible experience, do not recommend",
    "Poor quality and late delivery", "The support team was rude and useless",
    "Worst purchase I have made this year", "Complete waste of money",
    "I really disliked using this app", "Confusing design and very buggy",
    "Totally unsatisfied, one star", "What a dreadful surprise, hated it",
    "Slow and unreliable", "This fell far below my expectations",
    "Boring movie with a terrible story", "The food was cold and the staff unfriendly",
    "I'm furious about the results", "Dreadful performance, bad job",
    "Very upset, will never buy again",
]
NEUTRAL = [
    "The package arrived on Tuesday", "This is a phone with a 6 inch screen",
    "The meeting is scheduled for 3 pm", "It is a book about world history",
    "The store opens at nine in the morning", "I bought this item yesterday",
    "The report contains three sections", "The app is available on Android and iOS",
    "He walked to the station", "The package weighs two kilograms",
    "The movie lasts two hours", "Water boils at one hundred degrees Celsius",
    "The office is located on the second floor", "She sent the email this morning",
    "The device comes in black and white", "The class meets on Mondays",
    "The software was updated to version two", "There are five items in the cart",
    "The restaurant is near the train station", "The document has ten pages",
]


def build_dataset():
    texts = POSITIVE + NEGATIVE + NEUTRAL
    labels = ["positive"] * len(POSITIVE) + ["negative"] * len(NEGATIVE) + ["neutral"] * len(NEUTRAL)
    return texts, labels


def train(output_path: str) -> Pipeline:
    texts, labels = build_dataset()
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, C=10.0)),
        ]
    )
    pipeline.fit(texts, labels)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(pipeline, fh)
    return pipeline


if __name__ == "__main__":
    path = os.getenv("MODEL_PATH", "./model/sentiment_model.pkl")
    train(path)
    print(f"Model trained and saved to {path}")
