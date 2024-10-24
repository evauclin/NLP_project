import os
import click
import joblib
import pandas as pd

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
    default="src/data/raw/data.csv",
    help="File training data"
)
@click.option(
    "--model_dump_filename",
    default="src/model/dump.json",  # Updated to match your structure
    help="File to dump model"
)
@click.option(
    "--output_filename",
    default="src/data/processed/prediction.csv",
    help="Output file for predictions"
)

def predict(input_filename, model_dump_filename, output_filename):
    """
    Fait des prédictions sur de nouvelles données en utilisant le modèle entraîné.
    """
    # Assurer que le répertoire de sortie existe
    ensure_dir(output_filename)

    # Charger le modèle
    print(f"Chargement du modèle depuis {model_dump_filename}")
    model = joblib.load(model_dump_filename)

    # Charger et préparer les données de test
    print(f"Chargement des données depuis {input_filename}")
    df_test = pd.read_csv(input_filename)

    # Préparation des features pour la prédiction
    print("Préparation des features...")
    X_test = make_features(df_test, is_training=False)

    # Faire les prédictions
    print("Génération des prédictions...")
    predictions = model.predict(X_test)

    # Créer un DataFrame avec les prédictions
    results = pd.DataFrame({
        'id': df_test.index,
        'video_name': df_test['video_name'],
        'prediction': predictions
    })

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