# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download the 20 Newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)

# Preprocess the text data
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=nltk.word_tokenize)
X = newsgroups_train.data
y = newsgroups_train.target
X = [stemmer.stem(word) for word in X]
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Support Vector Machine (SVM) classifier
svm = LinearSVC()
svm.fit(X_train, y_train)

# Build the Streamlit application
st.title("Text Classifier App")
text = st.text_input("Enter a text:")
options = newsgroups_train.target_names
option = st.selectbox("Select a category:", options)
if st.button("Classify"):
    text = [stemmer.stem(word) for word in [text]]
    text = vectorizer.transform(text)
    result = svm.predict(text)
    st.write(f"The predicted category is: {options[result[0]]}")

