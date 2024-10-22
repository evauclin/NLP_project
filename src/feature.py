import subprocess
from spacy.lang.fr import stop_words
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import numpy as np


def process_text_lematization(text, nlp):
    print("download before loading")
    nlp = spacy.load('fr_core_news_md')
    print("download after loading")
    stop_words = nlp.Defaults.stop_words
    return " ".join([token.lemma_ for token in nlp(text)
                     if token.text.lower() not in stop_words and not token.is_punct])
def process_text_stemming(text, nlp):
    nlp = spacy.load('fr_core_news_md')
    stop_words = nlp.Defaults.stop_words
    return " ".join([token.lemma_ for token in nlp(text)
                     if token.text.lower() not in stop_words and not token.is_punct])

def apply_CountVectorizer(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def apply_word2vec(X_train, X_test):
    # Entraîner le modèle Word2Vec sur les phrases du jeu d'entraînement
    model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)

    # Fonction pour obtenir le vecteur moyen d'une phrase
    def get_sentence_vector(sentence, model):
        # Garder uniquement les mots présents dans le vocabulaire du modèle Word2Vec
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    # Transformer chaque phrase en vecteur
    X_train_vect = np.array([get_sentence_vector(sentence, model) for sentence in X_train])
    X_test_vect = np.array([get_sentence_vector(sentence, model) for sentence in X_test])

    return X_train_vect, X_test_vect


def make_features(df_train, df_test):
    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])

    nlp = spacy.load('fr_core_news_md')
    df_train["video_name_lematized"] = df_train["video_name"].apply(process_text_lematization)


    # Appliquer le même traitement sur le jeu de test
    df_test["video_name_lematized"] = df_test["video_name"].apply(process_text_lematization)

    y_train = df_train["label"]
    y_test = df_test["label"]
    print('data before vectorization')
    X_train , X_test = apply_word2vec(df_train["video_name_lematized"], df_test["video_name_lematized"])
    print('data after vectorization')

    print(X_test)

    return X_train, y_train
