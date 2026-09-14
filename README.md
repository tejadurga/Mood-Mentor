# 🧠 Mood Mentor

Mood Mentor is an NLP-based text analysis application developed as part of the **Infosys Springboard Virtual Internship**.

The project combines traditional sentiment analysis with Transformer-based emotion classification to analyze the emotional content of text.

---

## 📌 Project Overview

Mood Mentor analyzes user-provided text through two complementary approaches:

1. **VADER Sentiment Analysis**
2. **Transformer-based Emotion Classification**

The system first determines the overall sentiment of the text and then predicts one or more emotions using a fine-tuned BERT model.

The project also includes DistilBERT integration for model comparison and an external ISEAR benchmark for validation.

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