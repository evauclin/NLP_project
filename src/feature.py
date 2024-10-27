import subprocess

import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
nlp = spacy.load("fr_core_news_md")



def process_text_lematization(text):
    stop_words = nlp.Defaults.stop_words
    return " ".join(
        [
            token.lemma_.lower()
            for token in nlp(text)
            if token.text.lower() not in stop_words and not token.is_punct
        ]
    )


#
# def process_text_stemming(text):
#     # Initialize the stemmer
#     stemmer = PorterStemmer()
#     stop_words = set(nltk.corpus.stopwords.words('english'))  # Get English stop words
#
#     # Tokenize the text and apply stemming
#     stemmed_text = [
#         stemmer.stem(word.lower())  # Apply stemming to the token
#         for word in word_tokenize(text)  # Tokenize the text
#         if word.lower() not in stop_words and word.isalnum()  # Filter out stop words and non-alphanumeric tokens
#     ]
#
#     return " ".join(stemmed_text)


def apply_CountVectorizer(X_train, X_test, save_path="src/model/count_vectorizer.pkl"):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    joblib.dump(vectorizer, save_path)
    return X_train, X_test


def apply_TFIDFVectorizer(X_train, X_test, save_path="src/model/tfidf_vectorizer.pkl"):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    joblib.dump(vectorizer, save_path)
    return X_train, X_test


def load_vectorizer_and_transform(
    X_test, preprocessing_function, load_path="src/model/count_vectorizer.pkl"
):
    X_test_pre = X_test.apply(preprocessing_function)

    vectorizer = joblib.load(load_path)

    X_test_transformed = vectorizer.transform(X_test_pre)

    return X_test_transformed


def apply_word2vec(X_train, X_test):
    model = Word2Vec(
        sentences=X_train, vector_size=100, window=5, min_count=1, workers=4
    )

    def get_sentence_vector(sentence, model):
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    X_train_vect = np.array(
        [get_sentence_vector(sentence, model) for sentence in X_train]
    )
    X_test_vect = np.array(
        [get_sentence_vector(sentence, model) for sentence in X_test]
    )

    return X_train_vect, X_test_vect


def save_word2vec_model(model, save_path="src/model/word2vec.model"):
    model.save(save_path)


def load_word2vec_and_transform(X_test, load_path="src/model/word2vec.model"):
    model = Word2Vec.load(load_path)

    def get_sentence_vector(sentence, model):
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    X_test_vect = np.array(
        [get_sentence_vector(sentence, model) for sentence in X_test]
    )

    return X_test_vect


def apply_word2vec(X_train, X_test, save_path="src/model/word2vec.model"):
    model = Word2Vec(
        sentences=X_train, vector_size=100, window=5, min_count=1, workers=4
    )

    save_word2vec_model(model, save_path)

    def get_sentence_vector(sentence, model):
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    X_train_vect = np.array(
        [get_sentence_vector(sentence, model) for sentence in X_train]
    )
    X_test_vect = np.array(
        [get_sentence_vector(sentence, model) for sentence in X_test]
    )

    return X_train_vect, X_test_vect


def make_features(df_train, df_test, encoding):
    df_train["video_name_lematized"] = df_train["video_name"].apply(
        lambda x: process_text_lematization(x)
    )
    df_test["video_name_lematized"] = df_test["video_name"].apply(
        lambda x: process_text_lematization(x)
    )

    y_train = df_train["label"]
    if encoding == "word2vec":
        X_train_to_vec, X_test_to_vec = apply_word2vec(
            df_train["video_name_lematized"], df_test["video_name_lematized"]
        )
    elif encoding == "count_vectorizer":
        X_train_to_vec, X_test_to_vec = apply_CountVectorizer(
            df_train["video_name_lematized"], df_test["video_name_lematized"]
        )
    elif encoding == "tfidf":
        X_train_to_vec, X_test_to_vec = apply_TFIDFVectorizer(
            df_train["video_name_lematized"], df_test["video_name_lematized"]
        )

    return X_train_to_vec, y_train
