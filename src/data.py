import pandas as pd
from sklearn.model_selection import train_test_split

def make_dataset(filename):
    df = pd.read_csv(filename)
    X, y = df["video_name"].values, df["is_comic"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    df_train = pd.DataFrame({"video_name": X_train, "label": y_train})
    df_test = pd.DataFrame({"video_name": X_test, "label": y_test})

    df_train.to_csv("./../data/raw/train.csv", index=False)
    df_test.to_csv("./../data/raw/test.csv", index=False)

    return df_train
