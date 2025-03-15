# Import required libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import torch
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')

# Load the dataset
file_path = "data/final_scraped_lyrics.csv"  # Update the file path as needed
df = pd.read_csv(file_path)

# Drop missing values
df.dropna(subset=["Lyrics"], inplace=True)

# Load stopwords once
nltk_stopwords = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = " ".join([word for word in text.split() if word not in nltk_stopwords])  # Remove stopwords
    return text

# Apply cleaning function
df["Cleaned_Lyrics"] = df["Lyrics"].apply(clean_text)

# Sentiment Analysis using TextBlob
df["TextBlob_Sentiment"] = df["Cleaned_Lyrics"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Lexical Diversity (Unique words ratio)
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if len(words) > 0 else 0  # Avoid division by zero

df["Lexical_Diversity"] = df["Cleaned_Lyrics"].apply(lexical_diversity)

# **Step 1: Train a Word2Vec Model**
lyrics_sentences = [lyric.split() for lyric in df["Cleaned_Lyrics"].dropna()]
word2vec_model = Word2Vec(sentences=lyrics_sentences, vector_size=100, window=5, min_count=2, workers=4)

# Convert lyrics into Word2Vec embeddings (average word vectors for each song)
def get_word2vec_embedding(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(word_vectors) == 0:
        return np.zeros(100)  # Return zero vector if no words are found
    return np.mean(word_vectors, axis=0)

df["Word2Vec_Embedding"] = df["Cleaned_Lyrics"].apply(get_word2vec_embedding)

# **Step 2: Use Pretrained BERT Model for Lyrics**
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract CLS token embedding

df["BERT_Embedding"] = df["Cleaned_Lyrics"].apply(get_bert_embedding)

# **Step 3: Prepare Features for Machine Learning**
X_word2vec = np.stack(df["Word2Vec_Embedding"].values)
X_bert = np.stack(df["BERT_Embedding"].values)

# Combine Word2Vec, BERT, Sentiment, and Lexical Diversity
X_combined = np.hstack([
    X_word2vec,  # Word2Vec embeddings
    X_bert,  # BERT embeddings
    df[["TextBlob_Sentiment", "Lexical_Diversity"]].values  # Additional features
])

# **Create a fake "Success" column (for now, randomly assigned, can be updated with actual data)**
np.random.seed(42)  # For reproducibility
df["Success"] = np.random.choice([0, 1], size=len(df))  # 0 = Not Successful, 1 = Successful

# **Train/Test Split**
X_train, X_test, y_train, y_test = train_test_split(X_combined, df["Success"], test_size=0.2, random_state=42)

# **Function to Train & Evaluate Models**
def train_and_evaluate(model, model_name):
    print(f"\n🔹 Training {model_name}...\n")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 60)

# **Train 3 Additional Models with Word Embeddings**
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Support Vector Machine (SVM)": SVC(kernel='linear', probability=True)
}

for model_name, model in models.items():
    train_and_evaluate(model, model_name)

# **Word Cloud of Successful Lyrics**
successful_lyrics = " ".join(df[df["Success"] == 1]["Cleaned_Lyrics"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(successful_lyrics)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Most Common Words in Successful Lyrics", fontsize=14)
plt.show()
