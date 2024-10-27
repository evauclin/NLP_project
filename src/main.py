import os

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from data import make_dataset
from feature import (
    load_vectorizer_and_transform,
    load_word2vec_and_transform,
    make_features,
    process_text_lematization,
)
from models import make_model

# possible values: "word2vec" , "count_vectorizer" and tfidf
encoder = "word2vec"


def ensure_dir(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--input_filename", default="src/data/raw/data.csv", help="File training data"
)
@click.option(
    "--model_dump_filename",
    default="src/model/dump.json",  # Updated to match your structure
    help="File to dump model",
)
def train(input_filename, model_dump_filename):
    ensure_dir(model_dump_filename)
    df_train, df_test = make_dataset(input_filename)
    X_train, y_train = make_features(df_train, df_test, encoder)
    model = make_model()
    model.fit(X_train, y_train)
    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option(
    "--input_filename", default="src/data/raw/test.csv", help="File training data"
)
@click.option(
    "--model_dump_filename", default="src/model/dump.json", help="File to dump model"
)
@click.option(
    "--output_filename",
    default="src/data/processed/prediction.csv",
    help="Output file for predictions",
)
def predict(input_filename, model_dump_filename, output_filename):
    ensure_dir(output_filename)

    model = joblib.load(model_dump_filename)
    df_test = pd.read_csv(input_filename)

    if encoder == "word2vec":
        X_test_transformed = load_word2vec_and_transform(df_test["video_name"])
    if encoder == "count_vectorizer":
        X_test_transformed = load_vectorizer_and_transform(
            df_test["video_name"], process_text_lematization
        )
    if encoder == "tfidf":
        X_test_transformed = load_vectorizer_and_transform(
            df_test["video_name"], process_text_lematization
        )

    predictions = model.predict(X_test_transformed)

    results = pd.DataFrame(
        {"id": df_test.index, "prediction": predictions, "label": df_test.label}
    )

    print("Accuracy test : ", accuracy_score(results["prediction"], results["label"]))

    # Sauvegarder les prédictions
    print(f"Sauvegarde des prédictions dans {output_filename}")
    results.to_csv(output_filename, index=False)
    print("Prédictions terminées!")

    return results


@click.command()
@click.option(
    "--input_filename", default="src/data/raw/data.csv", help="File training data"
)
def evaluate(input_filename):
    df_train, df_test = make_dataset(input_filename)
    X_train, y_train = make_features(df_train, df_test, encoder)

    model = make_model()
    model.fit(X_train, y_train)

    return evaluate_model(model, X_train, y_train)


def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Scores pour chaque pli (fold): {scores}")
    print(f"Score moyen : {np.mean(scores)}")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
