import click
import joblib

from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/data.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    print("Loading data...")
    df_train, df_test = make_dataset(input_filename)
    print("Data loaded")
    X_train, y_train = make_features(df_train, df_test)
    print("Features created")

    model = make_model()
    model.fit(X_train, y_train)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/raw/test.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    model = joblib.load(model_dump_filename)
    pass


@click.command()
@click.option("--input_filename", default="data/raw/data.csv", help="File training data")
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
