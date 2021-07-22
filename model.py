import fasttext
import numpy as np
from skmultilearn.adapt import MLkNN, MLARAM
from collections import defaultdict


class FastText:
    """
    train , save and infer a skipgram model for word embeddings using fasttext lib.
    """

    def __init__(self, config=None, df=None, embedding_size=100) -> None:
        """
        args :
            config : config dict
            df : df containing the urls to train or to infer the labels from.
            embedding_size : embedding size for the skipgram word embedding model.
        """
        self.config = config
        self.df = df
        self.embedding_size = embedding_size

    def train(self):

        with open(self.config.training_data_fastext_path, "w") as f:
            for text in self.df["text_url"]:
                f.write(text + "\n")
        model = fasttext.train_unsupervised(
            self.config.training_data_fastext_path, model="skipgram"
        )
        # save here
        model.save_model(self.config.fast_text_path)

    def get_embeddings(self):
        """
        infer the embeddings for the object dataframe.
        """
        model = fasttext.load_model(self.config.fast_text_path)
        embeddings = []
        for text_url in self.df["text_url"]:
            split_text = text_url.split()
            weight = 1 / 2
            vect = np.zeros((100,))

            for word in split_text:
                vect += weight * model.get_word_vector(word)
                weight = weight * self.config.ratio
            embeddings.append(vect)

        return np.stack(embeddings, axis=0)


class RuleBased:
    def __init__(self, df_train, stopwords_list):
        self.df_train = df_train
        self.stopwords_list = stopwords_list
        self.word_to_label = self.get_word_to_label_dict()

    def get_word_to_label_dict(self):
        """
        create a dict that maps each word in the training set to the
        labels that co-occur with it , along with the frequency of co-occurence.
        """
        word_to_label_freq = defaultdict(lambda: defaultdict(int))
        for idx, rows in self.df_train.iterrows():
            split_words = rows["text_url"].split()
            for word in split_words:
                if word in self.stopwords_list:
                    continue
                for label in rows["labels"]:
                    word_to_label_freq[word][label] += 1

        word_to_label = defaultdict(lambda: defaultdict(float))
        for word, labels_dict in word_to_label_freq.items():
            word_to_label[word] = {
                k: v / sum(list(labels_dict.values()))
                for k, v in labels_dict.items()
            }
        word_to_label = dict(word_to_label)
        return word_to_label

    def predict(self, sentence, threshold=0.275):
        """
        Predict the URLs of a sentence.
        """
        split_sentence = sentence.split()
        labels_sentence = defaultdict(float)
        for word in split_sentence:
            if word not in self.word_to_label.keys():
                continue
            word_labels_dict = self.word_to_label[word]
            for label in word_labels_dict:
                labels_sentence[label] += word_labels_dict[label]
        sorted_labels = {
            k: v
            for k, v in sorted(
                labels_sentence.items(), key=lambda item: item[1], reverse=True
            )
        }
        normalized_sorted_labels = {
            k: v / sum(list(sorted_labels.values()))
            for k, v in sorted_labels.items()
        }
        if len(normalized_sorted_labels) == 0:
            return []
        sum_length, p = 0, 0
        items_dict = list(normalized_sorted_labels.items())
        returned_labels = []
        while sum_length < threshold:
            returned_labels.append(items_dict[p][0])
            sum_length += items_dict[p][1]
            p += 1
        return returned_labels


def get_classifier(config):
    if config.classifier_name == "mlknn":
        return MLkNN(k=config.mlknn_k)
    elif config.classifier_name == "mlaram":
        return MLARAM(
            threshold=config.thresh_mlaram, vigilance=config.vigilance
        )