from config import get_arguments
from load_data import LoadData
from model import FastText, get_classifier
from utils import get_one_hot_labels
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import numpy as np


def load_test_data(config):
    """
    load test data from config.
    """
    load_instance = LoadData.load(config.preprocessing_class_path)
    if config.preprocess:
        load_instance.df_data = pd.read_csv(config.path_test_data)
        df_test = load_instance.preprocess()
        return df_test
    else:
        config.n_classes = load_instance.config.n_classes
        df = pd.read_csv(config.path_test_data)
        df["labels"] = df["labels"].apply(eval)
        return df


def get_IoU_score(y_test, predictions):
    """
    give a target list y_test and the predictions of the multilabel classifier  ,
    this function returns the IoU score.
    """
    score = 0
    for target, pred in zip(y_test, predictions):
        target_ones = np.where(target == 1)[0]
        pred_ones = np.where(np.array(pred) == 1)[0]
        current_score = len(
            set(target_ones).intersection(set(pred_ones))
        ) / len(set(target_ones).union(set(pred_ones)))
        score += current_score

    return score / len(y_test)


if __name__ == "__main__":
    parser = get_arguments()
    config = parser.parse_args()
    df_test = load_test_data(config)
    fast_text = FastText(config, df_test)
    X_test = fast_text.get_embeddings()
    y_test = get_one_hot_labels(df_test, config)

    # load classifer
    classifier = pickle.load(open(config.model_path, "rb"))
    print('generating predictions ...')
    predictions = classifier.predict(X_test)
    print("Exact accuracy is {}".format(accuracy_score(y_test, predictions)))
    print("IoU metric score is {}".format(get_IoU_score(y_test, predictions)))
