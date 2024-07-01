import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
nltk.download('punkt')

# Sample data
data = {
    'text': [
        "This is a sample document.",
        "This document is a sample document.",
        "This is another example document.",
        "Completely different text here."
    ],
    'label': [0, 1, 0, 0]  # 0 for original, 1 for plagiarized
}

df = pd.DataFrame(data)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Cosine Similarity Matrix:")
print(cosine_sim)

def detect_plagiarism(similarity_matrix, threshold=0.8):
    n = similarity_matrix.shape[0]
    plagiarism_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] > threshold:
                plagiarism_pairs.append((i, j))

    return plagiarism_pairs

plagiarism_threshold = 0.8
plagiarized_docs = detect_plagiarism(cosine_sim, threshold=plagiarism_threshold)

print("Detected Plagiarized Document Pairs (indices):")
print(plagiarized_docs)
