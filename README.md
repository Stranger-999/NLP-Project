# SMS Spam & Phishing Detection Using TF-IDF and Naive Bayes

## Project Overview
This project implements an automated Natural Language Processing (NLP) pipeline to classify SMS text messages into legitimate (`ham`) or unsolicited/malicious (`spam`). The system uses standard text preprocessing, TF-IDF vectorization, and a Multinomial Naive Bayes classifier.

## Dataset
* **Source:** SMS Spam Collection Dataset (UCI ML Repository)
* **Total Samples:** 5,572 messages
* **Classes:** `ham` (legitimate) and `spam`

## Project Structure
```text
NLP-Project/
├── README.md
├── requirements.txt
├── source_code.py
├── dataset/
├── screenshots/
│   └── confusion_matrix.png
└── report/
    └── Project_Report.md