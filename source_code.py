"""
source_code.py
================
NLP-Based Sentiment Analysis of Product Reviews Using TF-IDF and Machine Learning

Pipeline:
    Raw Review Text -> Text Preprocessing -> TF-IDF Vectorization
        -> Classification Model (Naive Bayes / Logistic Regression / Linear SVM)
        -> Prediction -> Evaluation

Author : <your name here>
Course : <course name>
"""

import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans and normalizes a single review:
      1. Lowercase
      2. Remove punctuation and digits
      3. Tokenize (simple whitespace split)
      4. Remove stopwords
      5. Lemmatize each token
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["review_text", "sentiment"])
    df["clean_text"] = df["review_text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE MULTIPLE MODELS
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame, results_dir="results"):
    import os
    os.makedirs(results_dir, exist_ok=True)

    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorization (unigrams + bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    summary_rows = []
    best_model_name, best_f1, best_model_obj = None, -1, None

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="positive")
        rec = recall_score(y_test, preds, pos_label="positive")
        f1 = f1_score(y_test, preds, pos_label="positive")

        summary_rows.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-score": f1
        })

        print(f"\n===== {name} =====")
        print(classification_report(y_test, preds))

        cm = confusion_matrix(y_test, preds, labels=model.classes_)
        plot_confusion_matrix(cm, model.classes_, name, results_dir)

        if f1 > best_f1:
            best_f1, best_model_name, best_model_obj = f1, name, model

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{results_dir}/model_comparison.csv", index=False)
    print("\nModel comparison:\n", summary_df)

    plot_model_comparison(summary_df, results_dir)

    # Persist the best model + vectorizer for reuse
    joblib.dump(best_model_obj, f"{results_dir}/best_model.pkl")
    joblib.dump(vectorizer, f"{results_dir}/tfidf_vectorizer.pkl")
    print(f"\nBest model: {best_model_name} (F1={best_f1:.3f}) saved to {results_dir}/best_model.pkl")

    return summary_df, best_model_name, best_model_obj, vectorizer


def plot_confusion_matrix(cm, labels, model_name, results_dir):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fname = f"{results_dir}/confusion_matrix_{model_name.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_model_comparison(summary_df, results_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = range(len(summary_df))
    width = 0.2
    for i, metric in enumerate(metrics):
        ax.bar([p + i * width for p in x], summary_df[metric], width, label=metric)
    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_xticklabels(summary_df["Model"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{results_dir}/model_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. PREDICT ON NEW / UNSEEN REVIEWS
# ---------------------------------------------------------------------------
def predict_sentiment(text: str, model, vectorizer) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "dataset/product_reviews.csv"

    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    print(df[["review_text", "clean_text", "sentiment"]].head())

    print("\nTraining and evaluating models...")
    summary_df, best_name, best_model, vectorizer = train_and_evaluate(df)

    # -----------------------------------------------------------------
    # 5. EXTERNAL / OUT-OF-DOMAIN EVALUATION
    # -----------------------------------------------------------------
    # The in-domain test split (from the same generated templates) scores
    # very high, which can be misleading. To honestly evaluate generalization,
    # we test on a small hand-written set of reviews using vocabulary and
    # phrasing NOT present in the training templates (e.g. "useless",
    # "ripoff", "garbage", "mediocre").
    print("\n=== External hold-out evaluation (unseen vocabulary/phrasing) ===")
    ext_df = pd.read_csv("dataset/external_test_reviews.csv")
    ext_df["clean_text"] = ext_df["review_text"].apply(clean_text)
    ext_vec = vectorizer.transform(ext_df["clean_text"])
    ext_preds = best_model.predict(ext_vec)

    ext_acc = accuracy_score(ext_df["sentiment"], ext_preds)
    ext_f1 = f1_score(ext_df["sentiment"], ext_preds, pos_label="positive")
    print(f"External test accuracy: {ext_acc:.2f}  |  F1-score: {ext_f1:.2f}")

    for text, true, pred in zip(ext_df["review_text"], ext_df["sentiment"], ext_preds):
        mark = "OK " if true == pred else "XX "
        print(f"  [{mark}] true={true:<9} pred={pred:<9} | {text}")

    # Bar chart: in-domain vs external (out-of-domain) performance
    fig, ax = plt.subplots(figsize=(5, 4))
    in_domain_f1 = summary_df.loc[summary_df["Model"] == best_name, "F1-score"].values[0]
    ax.bar(["In-domain test set", "External hold-out set"], [in_domain_f1, ext_f1],
           color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title(f"Generalization Gap ({best_name})")
    for i, v in enumerate([in_domain_f1, ext_f1]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig("results/generalization_gap.png", dpi=150)
    plt.close(fig)

    with open("results/external_eval_summary.txt", "w") as f:
        f.write(f"External hold-out accuracy: {ext_acc:.2f}\n")
        f.write(f"External hold-out F1-score: {ext_f1:.2f}\n")
        f.write("\nMisclassified examples:\n")
        for text, true, pred in zip(ext_df["review_text"], ext_df["sentiment"], ext_preds):
            if true != pred:
                f.write(f"  true={true} pred={pred} | {text}\n")
