import os
import pandas as pd
import numpy as np
import tldextract
from urllib.parse import urlparse
from nltk.stem.snowball import FrenchStemmer
import re
import pickle

stemmer = FrenchStemmer()


class LoadData:
    """
    class for loading and preprocessing data
    """

    def __init__(self, config):
        """
        args :
            config : config file
        """
        self.config = config
        self.df_data = self.load_zip(config.path_zip_data)
        self.dict_mapping = self.get_mapping_target()
        self.config.n_classes = len(self.dict_mapping)
        self.stopwords_list = self.load_stop_words_list()

    def load_zip(self, path_data):
        """
        load zip files
        """
        df_data = pd.DataFrame(columns=["url", "target", "day"])
        for filename in os.listdir(path_data):
            if filename.endswith(".parquet"):
                df_data = pd.concat(
                    [
                        df_data,
                        pd.read_parquet(
                            os.path.join(path_data, filename), engine="pyarrow"
                        ),
                    ],
                    ignore_index=True,
                )
        return df_data

    def get_mapping_target(self):
        """
        Returns a dict that maps old targets to new ones having the property of being succesive.
        """
        full_labels = []
        for index, row in self.df_data.iterrows():
            labels = [label for label in row["target"].astype("int64")]
            full_labels += labels
        full_labels = np.array(full_labels)
        dict_mapping = dict(
            zip(np.unique(full_labels), range(len(np.unique(full_labels))))
        )
        return dict_mapping

    def load_stop_words_list(self):
        with open(self.config.path_stopwords_list, "r") as f:
            lines = f.readlines()
        lines = [l.replace("\n", "") for l in lines]
        return lines

    @staticmethod
    def get_new_target(target, dict_mapping):
        """
        map old targets to the new ones
        """
        labels = target.astype("int64")
        new_label = [dict_mapping[label] for label in labels]
        return new_label

    def parse_url(self, url):
        """
        parse the url. It returns the domain name and the words constituting the path. We remove
        stopwords , words with length less than 2 and digit parts. We finally stem the words.
        """

        def removing_condition(token, stopwords):
            cond = (
                any(c.isdigit() for c in token)
                or len(token) <= 2
                or token in stopwords
            )
            return not (cond)

        domain_name = tldextract.extract(url)[1]  ## extract domaine name
        full_path = urlparse(url).path
        ## split path based on special caracters
        first_tokens = re.split("[- _ % : , / \. \+ ]", full_path)
        ## get chars from tokens composed of chars + numbers. for instance , extract awd for
        ## awd789
        tokens = []
        for token in first_tokens:
            tokens += re.split("\d+", token)
        tokens = [
            stemmer.stem(token.lower())
            for token in tokens
            if removing_condition(token.lower(), self.stopwords_list)
        ]
        tokens = [
            token
            for token in tokens
            if removing_condition(token, self.stopwords_list)
        ]
        # return unique elements
        final_sentence = list(dict.fromkeys([domain_name] + tokens))
        return " ".join(final_sentence)

    def preprocess(self):
        """
        preprocessing function
        """
        ## first we preprocess the labels
        self.df_data["labels"] = self.df_data["target"].apply(
            lambda x: self.get_new_target(x, self.dict_mapping)
        )
        ## next we preprocess the urls
        self.df_data["text_url"] = self.df_data["url"].apply(
            lambda x: self.parse_url(x)
        )

    def save(self):
        pickle.dump(self, open(self.config.preprocessing_class_path, "wb"))

    @staticmethod
    def load(preprocessing_class_path):
        return pickle.load(open(preprocessing_class_path, "rb"))
