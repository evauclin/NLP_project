import subprocess
import joblib
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import FrenchStemmer
import nltk
import pandas as pd
from typing import Callable, Tuple, List

# Download required language model and stopwords
subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
nlp = spacy.load("fr_core_news_md")
nltk.download("stopwords")

# Initialize French stemmer
stemmer = FrenchStemmer("french")


def process_text_lemmatization(text: str) -> str:
    """
    Lemmatizes the input text and removes stop words and punctuation.

    Args:
        text (str): The text to process.

    Returns:
        str: A lemmatized and cleaned version of the input text.
    """
    stop_words = nlp.Defaults.stop_words
    return " ".join([
        token.lemma_.lower()
        for token in nlp(text)
        if token.text.lower() not in stop_words and not token.is_punct
    ])


def process_text_stemming(text: str) -> str:
    """
    Applies stemming to the input text and removes stop words and punctuation.

    Args:
        text (str): The text to process.

    Returns:
        str: A stemmed and cleaned version of the input text.
    """
    stop_words = nlp.Defaults.stop_words
    return " ".join([
        stemmer.stem(token.text.lower())
        for token in nlp(text)
        if token.text.lower() not in stop_words and not token.is_punct
    ])


def apply_count_vectorizer(X_train: List[str], X_test: List[str], save_path: str = "src/model/count_vectorizer.pkl") -> \
Tuple[np.ndarray, np.ndarray]:
    """
    Fits a CountVectorizer on training data and transforms both train and test data.

    Args:
        X_train (list of str): List of training text data.
        X_test (list of str): List of testing text data.
        save_path (str): Path to save the fitted CountVectorizer model.

    Returns:
        tuple: Transformed training and test data arrays.
    """
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    joblib.dump(vectorizer, save_path)
    return X_train, X_test


def apply_tfidf_vectorizer(X_train: List[str], X_test: List[str], save_path: str = "src/model/tfidf_vectorizer.pkl") -> \
Tuple[np.ndarray, np.ndarray]:
    """
    Fits a TfidfVectorizer on training data and transforms both train and test data.

    Args:
        X_train (list of str): List of training text data.
        X_test (list of str): List of testing text data.
        save_path (str): Path to save the fitted TfidfVectorizer model.

    Returns:
        tuple: Transformed training and test data arrays.
    """
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    joblib.dump(vectorizer, save_path)
    return X_train, X_test


def load_vectorizer_and_transform(X_test: pd.Series, preprocessing_function: Callable[[str], str],
                                  load_path: str = "src/model/count_vectorizer.pkl") -> np.ndarray:
    """
    Loads a saved vectorizer model and applies it to preprocessed test data.

    Args:
        X_test (pd.Series): Series of test text data.
        preprocessing_function (function): Preprocessing function to apply to X_test.
        load_path (str): Path to load the saved vectorizer model.

    Returns:
        np.ndarray: Transformed test data array.
    """
    X_test_pre = X_test.apply(preprocessing_function)
    vectorizer = joblib.load(load_path)
    X_test_transformed = vectorizer.transform(X_test_pre)
    return X_test_transformed


def apply_word2vec(X_train: List[List[str]], X_test: List[List[str]], save_path: str = "src/model/word2vec.model") -> \
Tuple[np.ndarray, np.ndarray]:
    """
    Trains a Word2Vec model on training data and transforms both train and test data.

    Args:
        X_train (list of list of str): List of tokenized sentences in training data.
        X_test (list of list of str): List of tokenized sentences in test data.
        save_path (str): Path to save the Word2Vec model.

    Returns:
        tuple: Word2Vec transformed training and test data arrays.
    """
    model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)
    model.save(save_path)

    def get_sentence_vector(sentence: List[str], model: Word2Vec) -> np.ndarray:
        """
        Computes the average vector for a given sentence based on a Word2Vec model.

        Args:
            sentence (list of str): List of words in the sentence.
            model (Word2Vec): Trained Word2Vec model.

        Returns:
            np.ndarray: Average vector for the sentence.
        """
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    X_train_vect = np.array([get_sentence_vector(sentence, model) for sentence in X_train])
    X_test_vect = np.array([get_sentence_vector(sentence, model) for sentence in X_test])
    return X_train_vect, X_test_vect


def load_word2vec_and_transform(X_test: List[List[str]], load_path: str = "src/model/word2vec.model") -> np.ndarray:
    """
    Loads a Word2Vec model and transforms test data using the model.

    Args:
        X_test (list of list of str): List of tokenized sentences in test data.
        load_path (str): Path to load the Word2Vec model.

    Returns:
        np.ndarray: Word2Vec transformed test data array.
    """
    model = Word2Vec.load(load_path)

    def get_sentence_vector(sentence: List[str], model: Word2Vec) -> np.ndarray:
        """
        Computes the average vector for a given sentence based on a Word2Vec model.

        Args:
            sentence (list of str): List of words in the sentence.
            model (Word2Vec): Trained Word2Vec model.

        Returns:
            np.ndarray: Average vector for the sentence.
        """
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    X_test_vect = np.array([get_sentence_vector(sentence, model) for sentence in X_test])
    return X_test_vect


def make_features(df_train: pd.DataFrame, df_test: pd.DataFrame, encoding: str) -> Tuple[np.ndarray, pd.Series]:
    """
    Generates features from text data based on specified encoding method.

    Args:
        df_train (pd.DataFrame): Training dataframe containing 'video_name' and 'label' columns.
        df_test (pd.DataFrame): Testing dataframe containing 'video_name' column.
        encoding (str): Encoding method to use, one of 'word2vec', 'count_vectorizer', or 'tfidf'.

    Returns:
        tuple: Encoded training data array and corresponding labels.
    """
    df_train["video_name_lematized"] = df_train["video_name"].apply(lambda x: process_text_lemmatization(x))
    df_test["video_name_lematized"] = df_test["video_name"].apply(lambda x: process_text_lemmatization(x))

    y_train = df_train["label"]

    if encoding == "word2vec":
        X_train_to_vec, X_test_to_vec = apply_word2vec(
            df_train["video_name_lematized"], df_test["video_name_lematized"]
        )
    elif encoding == "count_vectorizer":
        X_train_to_vec, X_test_to_vec = apply_count_vectorizer(
            df_train["video_name_lematized"], df_test["video_name_lematized"]
        )
    elif encoding == "tfidf":
        X_train_to_vec, X_test_to_vec = apply_tfidf_vectorizer(
            df_train["video_name_lematized"], df_test["video_name_lematized"]
        )

    return X_train_to_vec, y_train
