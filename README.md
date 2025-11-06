# 📡 Spam vs Ham SMS Classification using Machine Learning & Deep Learning

*End-to-end binary text classification leveraging Random Forest, Naïve Bayes & LSTM architectures for intelligent spam detection.*

---

## 🧭 Project Overview
Unsolicited spam messages threaten user privacy, waste bandwidth, and degrade user experience.  
This project builds **predictive models** capable of accurately distinguishing **spam** from **ham (legitimate)** SMS messages using both **classical ML** and **deep learning (LSTM)** techniques.

The pipeline covers:
- Comprehensive **text preprocessing** (cleaning → tokenization → stemming)  
- **Exploratory Data Analysis (EDA)** for data understanding  
- **Feature engineering** and encoding  
- Training + evaluation of **Random Forest**, **Naïve Bayes**, and **LSTM** models  
- Comparative insights on accuracy, F1-score, and ROC-AUC  

---

## 🎯 Problem Statement
> Build a **binary classifier** to label SMS messages as `spam` or `ham` using the **UCI SMS Spam Collection Dataset**.

---

## 🧱 System Architecture

      ┌───────────────────────────────┐
      │  Raw SMS Text Dataset         │
      └───────────────────────────────┘
                     │
                     ▼
          ┌────────────────────────┐
          │  Text Preprocessing    │
          │  • Cleaning            │
          │  • Tokenization        │
          │  • Stop-word Removal   │
          │  • Stemming            │
          └────────────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │  Feature Engineering & Encoding │
      └─────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │     Model Training & Evaluation        │
    │  • Random Forest                       │
    │  • Naïve Bayes                         │
    │  • Deep Learning (LSTM)                │
    └────────────────────────────────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │  Performance Metrics    │
         │  • Accuracy, Precision  │
         │  • Recall, F1, AUC      │
         └────────────────────────┘

---

## ⚙️ Implementation Workflow

### 1. **Data Preprocessing**
- Load dataset into a **pandas DataFrame**  
- Remove nulls, punctuation, and extra spaces  
- Tokenize messages into words  
- Remove English **stop words**  
- Apply **stemming** to normalize vocabulary  
- Encode target labels (`ham = 0`, `spam = 1`)  
- Split data into **train/test** partitions  

### 2. **Exploratory Data Analysis (EDA)**
- Pie chart: spam vs ham ratio  
- Histogram: message length distribution  
- Word-frequency analysis of top spam tokens  

### 3. **Model Development**
#### 🔹 Random Forest Classifier
- Ensemble of decision trees with cross-validation  
- Evaluated via accuracy, precision, recall, F1-score, ROC-AUC  

#### 🔹 Naïve Bayes Classifier
- Probabilistic baseline assuming feature independence  
- Quick training, strong recall but lower precision  

#### 🔹 Deep Learning LSTM
- Word embeddings + sequential learning  
- Layers: `Embedding → LSTM → Dense(sigmoid)`  
- Optimizer: Adam Loss: Binary Cross-Entropy  
- Trained over multiple epochs with accuracy/loss plots  

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| Language | **Python 3.11+** |
| Libraries | `pandas`, `numpy`, `sklearn`, `tensorflow/keras`, `matplotlib`, `seaborn` |
| Environment | Jupyter Notebook / Google Colab |
| Dataset | [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |

### 🔧 Installation
```bash
git clone https://github.com/MohanSaiBandarupalli/Final_Project
cd Final_Project
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn
