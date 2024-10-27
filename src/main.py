import os
import click
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from typing import List

from data import make_dataset
from feature import (
    load_vectorizer_and_transform,
    load_word2vec_and_transform,
    make_features,
    process_text_lemmatization,
    process_text_stemming
)
from models import make_model

# Possible values: "word2vec", "count_vectorizer", and "tfidf"
encoder = "count_vectorizer"


def ensure_dir(file_path: str) -> None:
    """
    Creates a directory if it doesn't already exist.

    Args:
        file_path (str): Path for the file whose directory is to be ensured.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


@click.group()
def cli():
    """
    Command Line Interface (CLI) for training, prediction, and evaluation of models.
    """
    pass


@click.command()
@click.option(
    "--input_filename", default="src/data/raw/train.csv", help="Path to the training data file"
)
@click.option(
    "--model_dump_filename",
    default="src/model/dump.json",
    help="Path to save the trained model"
)
def train(input_filename: str, model_dump_filename: str) -> None:
    """
    Trains a model using specified training data and saves the model to a file.

    Args:
        input_filename (str): Path to the training data file.
        model_dump_filename (str): Path where the model will be saved.
    """
    ensure_dir(model_dump_filename)
    df_train, df_test = make_dataset(input_filename)
    X_train, y_train = make_features(df_train, df_test, encoder)
    model = make_model()
    model.fit(X_train, y_train)
    joblib.dump(model, model_dump_filename)
    print(f"Model saved to {model_dump_filename}")


@click.command()
@click.option(
    "--input_filename", default="src/data/raw/test.csv", help="Path to the test data file"
)
@click.option(
    "--model_dump_filename", default="src/model/dump.json", help="Path to the saved model"
)
@click.option(
    "--output_filename",
    default="src/data/processed/prediction.csv",
    help="Path to save predictions"
)
def predict(input_filename: str, model_dump_filename: str, output_filename: str) -> None:
    """
    Loads a model and test data, performs predictions, and saves results to a file.

    Args:
        input_filename (str): Path to the test data file.
        model_dump_filename (str): Path to the saved model file.
        output_filename (str): Path to save prediction results.

    Returns:
        pd.DataFrame: DataFrame containing the predictions and actual labels.
    """
    ensure_dir(output_filename)
    model = joblib.load(model_dump_filename)
    df_test = pd.read_csv(input_filename)

    # Transform test data based on encoder type
    if encoder == "word2vec":
        X_test_transformed = load_word2vec_and_transform(df_test["video_name"])
    elif encoder == "count_vectorizer":
        X_test_transformed = load_vectorizer_and_transform(
            df_test["video_name"], process_text_lemmatization
        )
    elif encoder == "tfidf":
        X_test_transformed = load_vectorizer_and_transform(
            df_test["video_name"], process_text_lemmatization
        )
    else:
        raise ValueError(f"Invalid encoder type: {encoder}")

    # Predict and save results
    predictions = model.predict(X_test_transformed)
    results = pd.DataFrame(
        {"id": df_test.index, "prediction": predictions, "label": df_test.label}
    )

    accuracy = accuracy_score(results["prediction"], results["label"])
    print(f"Test Accuracy: {accuracy:.2f}")

    results.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")



@click.command()
@click.option(
    "--input_filename", default="src/data/raw/train.csv", help="Path to the data file for evaluation"
)
def evaluate(input_filename: str) -> List[float]:
    """
    Evaluates the model using cross-validation on the training data.

    Args:
        input_filename (str): Path to the training data file.

    Returns:
        list of float: Cross-validation scores for each fold.
    """
    df_train, df_test = make_dataset(input_filename)
    X_train, y_train = make_features(df_train, df_test, encoder)

    model = make_model()
    model.fit(X_train, y_train)
    scores = evaluate_model(model, X_train, y_train)
    return scores


def evaluate_model(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    Evaluates the model with cross-validation.

    Args:
        model: The model to evaluate.
        X (np.ndarray): Feature array.
        y (np.ndarray): Labels array.
        cv (int): Number of cross-validation folds.

    Returns:
        list of float: Cross-validation scores for each fold.
    """
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Scores per fold: {scores}")
    print(f"Mean score: {np.mean(scores):.2f}")


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
