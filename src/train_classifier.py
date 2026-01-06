import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_classifier(feature_matrix, labels):
    """
    Trains a lightweight classifier on aggregated evidence features.
    """

    X = np.array(feature_matrix)
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print(f"Validation Accuracy: {acc:.4f}")

    return clf