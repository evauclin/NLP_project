import os
import click
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from data import make_dataset
from feature import make_features
from models import make_model


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
    "--input_filename",
    default="src/data/raw/data.csv",
    help="File training data"
)
@click.option(
    "--model_dump_filename",
    default="src/model/dump.json",  # Updated to match your structure
    help="File to dump model"
)
def train(input_filename, model_dump_filename):
    ensure_dir(model_dump_filename)
    df_train, df_test = make_dataset(input_filename)
    X_train, y_train = make_features(df_train, df_test)
    model = make_model()
    model.fit(X_train, y_train)
    return joblib.dump(model, model_dump_filename)

@click.command()
@click.option(
    "--input_filename",
    default="src/data/raw/test.csv",
    help="File training data"
)
@click.option(
    "--model_dump_filename",
    default="src/model/dump.json",
    help="File to dump model"
)
@click.option(
    "--output_filename",
    default="src/data/processed/prediction.csv",
    help="Output file for predictions"
)

def predict(input_filename, model_dump_filename, output_filename):

    ensure_dir(output_filename)


    print(f"Chargement du modèle depuis {model_dump_filename}")
    model = joblib.load(model_dump_filename)


    print(f"Chargement des données depuis {input_filename}")
    df_test = pd.read_csv(input_filename)

    print("Préparation des features...")
    X_test = make_features(df_test)
    #X_test = np.load("src/data/raw/X_test.npy",allow_pickle=True)
    # Faire les prédictions
    print("Génération des prédictions...")
    predictions = model.predict(X_test)

    results = pd.DataFrame({
        'id': df_test.index,
        'prediction': predictions,
        'label': df_test.label
    })

    print("Accuracy test : ", accuracy_score(results['prediction'], results['label']))

    # Sauvegarder les prédictions
    print(f"Sauvegarde des prédictions dans {output_filename}")
    results.to_csv(output_filename, index=False)
    print("Prédictions terminées!")

    return results

@click.command()
@click.option(
    "--input_filename",
    default="src/data/raw/data.csv",
    help="File training data"
)
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)
    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df)
    # Object with .fit, .predict methods
    model = make_model()
    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)

def evaluate_model(model, X, y):
    # Run k-fold cross validation. Print results
    pass

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()