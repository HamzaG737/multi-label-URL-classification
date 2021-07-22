from eval import load_test_data
from config import get_arguments
from load_data import LoadData
from model import FastText, get_classifier
from utils import get_one_hot_labels

from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import pandas as pd


def split_data(df_data, config, test_frac=0.2):
    """
    split df_data to train and test.
    """
    df_train, df_test = train_test_split(df_data, test_size=test_frac)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_train.to_csv(config.path_train_data, index=False)
    df_test.to_csv(config.path_test_data, index=False)
    return df_train


def get_training_data(config):
    if os.path.isfile(config.path_train_data):
        return pd.read_csv(config.path_train_data)
    else:
        preprocessing_instance = LoadData(config)
        ## load data
        preprocessing_instance.preprocess()
        ## save preprocessing instance
        preprocessing_instance.save()

        ## split data
        df_train = split_data(preprocessing_instance.df_data, config)

        return df_train


if __name__ == "__main__":

    parser = get_arguments()
    config = parser.parse_args()
    config.eval = False

    print("processing data ...")
    df_train = get_training_data(config)

    # load fasttext and train it
    fast_text = FastText(config, df_train)
    fast_text.train()
    X_train = fast_text.get_embeddings(df_train)
    y_train = get_one_hot_labels(df_train, config)

    classifier = get_classifier(config)
    print("fitting classifier ...")
    classifier.fit(X_train, y_train)
    print("saving classifier ....")
    pickle.dump(classifier, open(config.model_path, "wb"))
