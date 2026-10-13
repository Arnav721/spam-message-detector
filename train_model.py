
import io
import urllib.request
import zipfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_FILE = MODEL_DIR / "spam_classifier.joblib"


def get_dataset():
    if DATA_FILE.exists():
        return

    print("Downloading the SMS Spam Collection dataset...")
    try:
        request = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            member = next(
                name for name in archive.namelist()
                if name.endswith("SMSSpamCollection")
            )
            DATA_FILE.write_bytes(archive.read(member))

    except Exception as error:
        print(f"Dataset download failed: {error}")
        print("Check your internet connection and run again.")
        raise


def main():
    get_dataset()

    data = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    data = data.dropna()
    data = data.drop_duplicates(subset="message")
    data = data[data["label"].isin(["ham", "spam"])]

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
)),
       ("classifier", LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)),
    ])

    print("Training the model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nDataset messages: {len(data)}")
    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["ham", "spam"],
        display_labels=["Not Spam", "Spam"],
        cmap="Blues",
    )
    plt.title("Spam Message Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved model to: {MODEL_FILE}")
    print(f"Saved confusion matrix to: {BASE_DIR / 'confusion_matrix.png'}")
    print("Training complete!")


if __name__ == "__main__":
    main()
