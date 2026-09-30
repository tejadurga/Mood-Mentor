# 🧠 Mood Mentor

Mood Mentor is an NLP-based text analysis application developed as part of the **Infosys Springboard Virtual Internship**.

The project combines traditional sentiment analysis with Transformer-based emotion classification to analyze text and produce personalized wellness recommendations.

---

## 📌 Project Overview

Mood Mentor analyzes user-provided text through two complementary approaches:

1. **VADER Sentiment Analysis**
2. **Transformer-based Emotion Classification**

The system determines the overall sentiment, predicts one or more emotions using a fine-tuned BERT model, analyzes the user's emotional state and trends, and ranks relevant wellness content.

The project also includes DistilBERT integration for model comparison, an external ISEAR benchmark for validation, explainable recommendations, and persistent recommendation feedback.

---

## 🎯 Project Objectives

The main objectives of Mood Mentor are:

- Analyze text sentiment as Positive, Negative, or Neutral.
- Detect six target emotions:
  - Joy
  - Sadness
  - Anger
  - Fear
  - Surprise
  - Disgust
- Support multi-label emotion classification.
- Generate model confidence scores.
- Compare BERT and DistilBERT performance.
- Validate the trained model using an external ISEAR benchmark.
- Analyze emotional intensity and historical emotion trends.
- Generate hybrid, personalized wellness recommendations.
- Explain recommendation scores and learn from user feedback.
- Integrate the complete analysis workflow into a Streamlit application.

---

# 🛠️ Technologies Used

## Programming Language

- Python 3.12

## NLP and Data Processing

- NLTK
- Pandas
- VADER Sentiment Analyzer

## Deep Learning

- PyTorch

## Transformer Models

- Hugging Face Transformers
- BERT (`bert-base-uncased`)
- DistilBERT (`distilbert-base-uncased`)

## Datasets

- GoEmotions
- ISEAR

## Application

- Streamlit

---

# 🏗️ System Architecture

The overall Mood Mentor workflow is:

```text
User Input
    ↓
Text Ingestion
    ↓
Text Preprocessing
    ↓
VADER Sentiment Analysis
    ↓
BERT / DistilBERT Emotion Analysis
    ↓
Six Emotion Probabilities
    ↓
Multi-Label Emotion Detection
    ↓
Primary Emotion
    ↓
Confidence Scores
    ↓
Final Analysis Result
```

The recommendation workflow combines user preferences, emotion relevance, metadata similarity, semantic matching, collaborative signals, historical trends, and feedback signals. Each recommendation includes score details that explain why it was selected.

---

# Getting Started

## Requirements

- Python 3.12 or a compatible Python 3 release
- The model files included in `models/bert_emotion` and `models/distilbert_emotion`

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

The Streamlit interface supports demo user profiles, text analysis, emotion history, trend visualization, recommendation ranking, recommendation explanations, and feedback recording.

## Run tests

```bash
python -m pytest -q
```

The tests cover emotion state and trend analysis, recommendation data and ranking, semantic matching, explainability, feedback persistence, and the integrated machine-learning workflow.

---

# Repository Structure

```text
app.py                         Streamlit application entry point
data/                          Demo users, histories, content, and evaluations
models/                        BERT and DistilBERT model artifacts
src/emotion_state.py           Emotional-state and intensity analysis
src/emotion_trends.py          Historical emotion trend analysis
src/recommendation_engine.py   Hybrid recommendation generation
src/recommendation_ranking.py  Recommendation scoring and ranking
src/recommendation_rules.py    Rule-based personalization signals
src/recommendation_content.py  Content-based matching
src/semantic_matching.py       Text and wellness-content similarity
src/recommendation_feedback.py Feedback persistence and aggregation
src/recommendation_explainability.py
                               Recommendation score explanations
src/test_*.py                  Unit and workflow tests
```

---

# Data and Models

The `data/` directory contains small demo CSV files used by the application, including user profiles, user history, emotion history, wellness content, recommendation evaluations, and feedback records.

The application loads the local BERT model from `models/bert_emotion`. DistilBERT artifacts are available for evaluation and model comparison. Training and validation utilities are located in `src/bert_training.py`, `src/distilbert_training.py`, and the related evaluation scripts.

---

# Evaluation Utilities

The repository includes scripts for:

- BERT and DistilBERT prediction and confidence validation.
- Emotion dataset preparation and validation.
- Recommendation evaluation and ranking analysis.
- Dataset inspection and report generation.

These scripts can be run from the repository root with Python, for example:

```bash
python src/evaluate_recommendations.py
```
