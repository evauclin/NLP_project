import subprocess
from spacy.lang.fr import stop_words
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import numpy as np
import pandas as pd


def process_text_lematization(text , nlp):
    stop_words = nlp.Defaults.stop_words
    return " ".join([token.lemma_.lower() for token in nlp(text)
                     if token.text.lower() not in stop_words and not token.is_punct])
def process_text_stemming(text, nlp):
    stop_words = nlp.Defaults.stop_words
    return " ".join([token.lemma_.lower() for token in nlp(text)
                     if token.text.lower() not in stop_words and not token.is_punct])

def apply_CountVectorizer(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test

def apply_word2vec(X_train, X_test):

    model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)


    def get_sentence_vector(sentence, model):

        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    X_train_vect = np.array([get_sentence_vector(sentence, model) for sentence in X_train])
    X_test_vect = np.array([get_sentence_vector(sentence, model) for sentence in X_test])

    return X_train_vect, X_test_vect


def make_features(df_input, df_test=None):
    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
    nlp = spacy.load('fr_core_news_md')

    df_input["video_name_lematized"] = df_input["video_name"].apply(lambda x: process_text_lematization(x, nlp))
    df_test["video_name_lematized"] = df_test["video_name"].apply(lambda x: process_text_lematization(x, nlp))
    y_train = df_input["label"]
    X_train_to_vec, X_test_to_vec = apply_word2vec(df_input["video_name_lematized"], df_test["video_name_lematized"])
    #X_train_to_vec, X_test_to_vec = apply_CountVectorizer(df_input["video_name_lematized"], df_test["video_name_lematized"])
    np.save("src/data/raw/X_train.npy", X_train_to_vec)
    np.save("src/data/raw/X_test.npy", X_test_to_vec)
    return X_train_to_vec, y_train


