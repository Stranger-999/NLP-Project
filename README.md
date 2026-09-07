# NLP-Based Sentiment Analysis of Product Reviews Using TF-IDF and Machine Learning

A mini NLP project that classifies product reviews as **positive** or **negative** using
TF-IDF feature extraction and classical machine learning (Naive Bayes, Logistic
Regression, Linear SVM).

## Highlights

- Full pipeline: preprocessing → TF-IDF → classification → evaluation
- Compares 3 ML algorithms with accuracy/precision/recall/F1 + confusion matrices
- **Includes an honest generalization check**: an independent, hand-written hold-out set
  (with vocabulary never seen in training) is used to test real-world performance,
  exposing a meaningful gap between in-domain accuracy and real generalization — see
  `report/Project_Report.md` Section 7–8 for the full analysis.

## Project Structure

```
NLP-Project/
│
├── README.md                  ← you are here
├── source_code.py             ← main pipeline (preprocessing, training, evaluation)
├── requirements.txt
├── dataset/
│   ├── generate_dataset.py    ← generates the self-created training dataset
│   ├── product_reviews.csv    ← 672 labeled training/test reviews
│   └── external_test_reviews.csv  ← 20 hand-written hold-out reviews (unseen vocabulary)
├── screenshots/                ← charts produced by the pipeline
│   ├── model_comparison.png
│   ├── confusion_matrix_*.png
│   └── generalization_gap.png
└── report/
    └── Project_Report.md       ← full project report (problem, methodology, results, limitations)
```

## How to Run

```bash
pip install -r requirements.txt
python source_code.py
```

This will:
1. Load and preprocess `dataset/product_reviews.csv`
2. Train Naive Bayes, Logistic Regression, and Linear SVM on TF-IDF features
3. Print classification reports and save confusion matrices / comparison charts to `results/`
4. Evaluate the best model on the independent external hold-out set and print/report the
   generalization gap
5. Save the trained model + vectorizer (`results/best_model.pkl`, `results/tfidf_vectorizer.pkl`)

## Key Result

| Evaluation | F1-score |
|---|---|
| In-domain test split | 1.00 |
| External hold-out (unseen vocabulary) | 0.83 |

The gap between these two numbers is the most important finding of this project — see the
report for full discussion.

## Author

<your name> — <course/institution>
