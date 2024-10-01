import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('stopwords')

def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    cleaned_texts = [" ".join([word for word in text.lower().split() if word not in stop_words]) for text in texts]
    return cleaned_texts

def compute_embeddings(texts):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()
    return embeddings, vectorizer.get_feature_names_out()

def find_top_word_pairs(embeddings, feature_names):
    cosine_sim_matrix = cosine_similarity(embeddings)
    top_word_pairs = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            similarity = cosine_sim_matrix[i][j]
            top_word_pairs.append((feature_names[i], feature_names[j], similarity))

    top_word_pairs.sort(key=lambda x: x[2], reverse=True)
    return top_word_pairs[:100]

# Example usage in main.py
if __name__ == "__main__":
    texts = [...]  # Your dataset texts
    cleaned_texts = preprocess_text(texts)
    embeddings, feature_names = compute_embeddings(cleaned_texts)
    top_word_pairs = find_top_word_pairs(embeddings, feature_names)
    print(top_word_pairs)
