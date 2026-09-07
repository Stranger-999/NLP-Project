import os
import re
import string
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK resources are available
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
  """Cleans, tokenizes, removes stopwords, and stems input text."""
  text = text.lower()
  text = re.sub(r"\d+", "", text)
  text = text.translate(str.maketrans("", "", string.punctuation))
  tokens = text.split()
  cleaned_tokens = [
      stemmer.stem(word)
      for word in tokens
      if word not in stop_words and len(word) > 2
  ]
  return " ".join(cleaned_tokens)


def load_data(filepath: str = "dataset/spam.csv") -> pd.DataFrame:
  """Loads the dataset from local file or downloads it directly if missing."""
  if not os.path.exists(filepath):
    print("Local dataset not found. Downloading SMS Spam Collection dataset...")
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
  else:
    df = pd.read_csv(
        filepath, encoding="latin-1", usecols=[0, 1], names=["label", "message"]
    )
  return df


def main():
  # 1. Load Data
  df = load_data()
  print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

  # 2. Preprocess Data
  print("Preprocessing text data...")
  df["cleaned_message"] = df["message"].apply(preprocess_text)
  df["target"] = df["label"].map({"ham": 0, "spam": 1})

  # 3. Train-Test Split (80/20 stratified)
  X_train, X_test, y_train, y_test = train_test_split(
      df["cleaned_message"],
      df["target"],
      test_size=0.2,
      random_state=42,
      stratify=df["target"],
  )

  # 4. Feature Extraction: TF-IDF
  tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
  X_train_vec = tfidf.fit_transform(X_train)
  X_test_vec = tfidf.transform(X_test)

  # 5. Model Training: Multinomial Naive Bayes
  model = MultinomialNB()
  model.fit(X_train_vec, y_train)

  # 6. Evaluation
  y_pred = model.predict(X_test_vec)
  print("\n--- Model Evaluation Report ---")
  print(
      classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"])
  )

  # 7. Confusion Matrix Plot
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(5, 4))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=["Ham", "Spam"],
      yticklabels=["Ham", "Spam"],
  )
  plt.title("SMS Classification Confusion Matrix")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.tight_layout()
  os.makedirs("screenshots", exist_ok=True)
  plt.savefig("screenshots/confusion_matrix.png")
  print("Confusion matrix saved to screenshots/confusion_matrix.png")

  # 8. Interactive Inference Sample
  sample_messages = [
      "Hey, are we still meeting in the cafeteria at 2 PM?",
      "URGENT! You have won a 1000 cash prize! Claim your reward now by texting WIN to 88882.",
  ]

  print("\n--- Sample Inferences ---")
  for msg in sample_messages:
    cleaned = preprocess_text(msg)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"Message: '{msg}'\nPrediction: {label} (Confidence: {prob:.2f})\n")


if __name__ == "__main__":
  main()