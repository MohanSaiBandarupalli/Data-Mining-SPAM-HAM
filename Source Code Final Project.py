import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

df=pd.read_csv("sms_spam_ham_dataset.csv",encoding="latin-1")

df = df.dropna(how="any", axis=1)
df.head()

#Finding length of each Message
df['text_len'] = df['Message'].apply(lambda x: len(x.split(' ')))
df.head()

#Encoding Spam-1 and Ham-0
le = LabelEncoder()
le.fit(df['Label'])

df['Label_Encoded'] = le.transform(df['Label'])
df.head()

#Finding count of ham and spam messages
df['Label'].value_counts()

labels = df['Label'].value_counts().index
sizes = df['Label'].value_counts().values

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Ham and Spam Messages')
plt.show()

text_lengths=df['text_len']
plt.figure(figsize=(10, 6))
sns.histplot(text_lengths, bins=50, kde=True)
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths')
plt.show()

"""# Data PreProcessing"""

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['Message_Clean'] = df['Message'].apply(clean_text)
df.head()

#Removing Stop Words
nltk.download('stopwords')
stop_words = stopwords.words('english')
more_stopwords = ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

df['Message_Clean'] = df['Message_Clean'].apply(remove_stopwords)
df.head()

#Stemming
stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text
df['Message_Clean'] = df['Message_Clean'].apply(stemm_text)
df.head()

x = df['Message_Clean']
y = df['Label_Encoded']

print(len(x), len(y))

"""# Modelling and Embedding"""

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# Assuming x_train and x_test are lists of tokenized documents
# Preprocess and tokenize your text data here

# Train or load a pre-trained Word2Vec model
word2vec_model = Word2Vec(sentences=x_train, vector_size=100, window=5, min_count=1)

# Generate word embeddings for each document
def document_embedding(document, model):
    vectors = [model.wv[word] for word in document if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

x_train_w2v = np.array([document_embedding(doc, word2vec_model) for doc in x_train])
x_test_w2v = np.array([document_embedding(doc, word2vec_model) for doc in x_test])

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # True Skill Statistic (TSS)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    tss = sensitivity + specificity - 1

    # Heidke Skill Score (HSS)
    n = tp + tn + fp + fn
    hss = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss=np.round(hss,2)
    tss=np.round(tss,2)
    accuracy=accuracy_score(y_true,y_pred).round(2)
    precision = precision_score(y_true,y_pred).round(2)
    recall = recall_score(y_true,y_pred).round(2)
    f1 = f1_score(y_true,y_pred).round(2)


    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'TSS': tss, 'HSS': hss,'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1-Score':f1}

"""# Random Forest Classifier"""

rfc = RandomForestClassifier(n_estimators=50, random_state=2)
rfc.fit(x_train_w2v, y_train)

# Predict on test data
y_pred = rfc.predict(x_test_w2v)
accuracy_score(y_test,y_pred)

y_train=y_train.to_numpy()

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix on Test Data using Random Forest Algorithm')
plt.show()

y_proba = rfc.predict_proba(x_test_w2v)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()

# Define the number of folds
num_folds = 10

# Initialize KFold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# Lists to store evaluation metrics for each fold
fold_metrics = []

# Iterate over each fold
for fold, (train_index, test_index) in enumerate(kf.split(x_train_w2v), 1):
    X_train, X_test = x_train_w2v[train_index], x_train_w2v[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    # Initialize and train the RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=50, random_state=2)
    rfc.fit(X_train, y_train_fold)

    # Predict on the test set
    y_pred_fold = rfc.predict(X_test)
    metrics=calculate_metrics(y_test_fold, y_pred_fold)

    # Store metrics for this fold
    fold_metrics.append(metrics)
# Create DataFrame from fold_metrics list
df_results_Random_Forest = pd.DataFrame(fold_metrics)

# Display the results
df_results_Random_Forest

# Average of 10 Folds execution
drf={}
for i in df_results_Random_Forest.columns:
  drf[i]=df_results_Random_Forest[i].mean().round(2)
print("Average of 10 Folds")
print(drf)

df_results_Random_Forest.to_csv("df_results_Random_Forest.csv")

"""# Naive Bayes"""

nb = GaussianNB()
nb.fit(x_train_w2v, y_train)
y_pred = nb.predict(x_test_w2v)
accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix on Test Data using Naive Bayes Algorithm')
plt.show()

y_proba = nb.predict_proba(x_test_w2v)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Naive Bayes')
plt.legend(loc='lower right')
plt.show()

num_folds = 10

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_metrics = []
for fold, (train_index, test_index) in enumerate(kf.split(x_train_w2v), 1):
    X_train, X_test = x_train_w2v[train_index], x_train_w2v[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    nb = GaussianNB()
    nb.fit(X_train, y_train_fold)
    y_pred_fold = nb.predict(X_test)
    metrics=calculate_metrics(y_test_fold, y_pred_fold)
    fold_metrics.append(metrics)
# Create DataFrame from fold_metrics list
df_results_Naive_Bayes = pd.DataFrame(fold_metrics)

# Display the results
df_results_Naive_Bayes

# Average of 10 Folds execution
dnb={}
for i in df_results_Naive_Bayes.columns:
  dnb[i]=df_results_Naive_Bayes[i].mean().round(2)
print("Average of 10 Folds")
print(dnb)

df_results_Naive_Bayes.to_csv("df_results_Naive_Bayes.csv")

"""# Deep Learning"""

#Preprocessing Data
max_words = 10000
max_len = 100  # Maximum sequence length

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)

# Define LSTM model
embedding_dim = 100
lstm_units = 128

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=lstm_units))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train_pad, np.array(y_train), epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test_pad, np.array(y_test))
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Plot accuracy curve
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

#Confusion Matrix
# Compute confusion matrix
y_pred_val = model.predict(x_test_pad)

    # Convert predicted probabilities to binary predictions
y_pred_val_binary = (y_pred_val > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_val_binary)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix on Test Data using Deep-Learning LSTM')
plt.show()

y_score = model.predict(x_test_pad)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Deep Learning LSTM')
plt.legend(loc='lower right')
plt.show()

# Perform cross-validation

# Initialize KFold cross-validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store metrics for each fold
all_metrics = []

for train_index, val_index in kf.split(x_train_pad, y_train):
    x_train_cv, x_val = x_train_pad[train_index], x_train_pad[val_index]
    y_train_cv, y_val = y_train[train_index], y_train[val_index]

    # Train the model
    model.fit(x_train_cv, y_train_cv, epochs=10, batch_size=64, verbose=0)

    # Evaluate the model on validation data
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    y_pred_val = model.predict(x_val)

    # Convert predicted probabilities to binary predictions
    y_pred_val_binary = (y_pred_val > 0.5).astype(int)
    metrics["loss"]=loss
    metrics = calculate_metrics(y_val, y_pred_val_binary)
    all_metrics.append(metrics)

# Calculate average metrics across all folds
df_results_Deep_Learning_LSTM = pd.DataFrame(all_metrics)

df_results_Deep_Learning_LSTM.to_csv("df_results_Deep_Learning_LSTM.csv")

df_results_Deep_Learning_LSTM

# Average of 10 Folds execution
dlstm={}
for i in df_results_Deep_Learning_LSTM.columns:
  dlstm[i]=df_results_Deep_Learning_LSTM[i].mean().round(2)
print("Average of 10 Folds")
print(dlstm)

# Assuming drf, dnb, and dlstm are defined somewhere in your code and hold different results
df_random_forest = pd.DataFrame([drf], index=['Random Forest'])
df_naive_bayes = pd.DataFrame([dnb], index=['Naive Bayes'])
df_lstm = pd.DataFrame([dlstm], index=['Deep Learning LSTM'])

# Concatenate all the dataframes into a single one for comparison
df_comparison = pd.concat([df_random_forest, df_naive_bayes, df_lstm])

# Display the comparison DataFrame
print("Comparison of Average Results Across 10 Folds:")
print(df_comparison)



