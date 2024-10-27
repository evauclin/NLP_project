import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def make_dataset(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads a CSV file containing video data, splits it into training and testing datasets,
    and saves these splits to separate CSV files.

    Args:
        filename (str): Path to the input CSV file containing 'video_name' and 'is_comic' columns.

    Returns:
        tuple: DataFrames for training and testing datasets, each with 'video_name' and 'label' columns.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Split features and labels
    X, y = df["video_name"].values, df["is_comic"].values

    # Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create DataFrames for training and testing sets
    df_train = pd.DataFrame({"video_name": X_train, "label": y_train})
    df_test = pd.DataFrame({"video_name": X_test, "label": y_test})

    # Save the splits to CSV files
    df_train.to_csv("src/data/raw/train.csv", index=False)
    df_test.to_csv("src/data/raw/test.csv", index=False)

    return df_train, df_test
