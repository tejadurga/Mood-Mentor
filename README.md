# 🧠 Mood Mentor

Mood Mentor is an NLP and Transformer-based emotional analysis and wellness recommendation application developed as part of the **Infosys Springboard Virtual Internship**.

The application analyzes user text, identifies sentiment and emotions, determines emotional intensity, analyzes historical trends, and provides personalized wellness recommendations.

---

## 🎯 Objectives

- Analyze text using VADER sentiment analysis.
- Detect six emotions using BERT:
  - Joy
  - Sadness
  - Anger
  - Fear
  - Surprise
  - Disgust
- Support multi-label emotion detection.
- Calculate emotion confidence and intensity.
- Analyze emotional trends.
- Generate personalized wellness recommendations.
- Rank recommendations dynamically.
- Provide recommendation explanations.
- Store recommendation history and feedback.
- Provide dashboard visualizations.
- Export reports in CSV and PDF.
- Support Streamlit and CLI interfaces.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| UI | Streamlit |
| NLP | NLTK, VADER |
| Deep Learning | PyTorch |
| Transformers | Hugging Face Transformers |
| Primary Model | BERT (`bert-base-uncased`) |
| Comparison Model | DistilBERT |
| Semantic Matching | MiniLM |
| Data Processing | Pandas, NumPy |
| Datasets | GoEmotions, ISEAR |
| Reporting | ReportLab |
| Development | VS Code, Git, GitHub |

---

## 🏗️ System Workflow

```text
User Input
   ↓
Text Ingestion
   ↓
Preprocessing
   ↓
VADER Sentiment
   ↓
BERT Emotion Classification
   ↓
Emotional State & Intensity
   ↓
Historical Trend Analysis
   ↓
Hybrid Recommendation Engine
   ↓
Dynamic Ranking
   ↓
Explainability
   ↓
History & Feedback
   ↓
Dashboard / Reports
