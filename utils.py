import numpy as np


def extract_one_hots(targets, config):
    one_hot_targets = np.zeros(config.n_classes)
    one_hot_targets[targets] = 1
    return one_hot_targets


def get_one_hot_labels(df, config):
    one_hot_labels = []
    for index, row in df.iterrows():
        one_hot_labels.append(extract_one_hots(row["labels"], config))
    one_hot_labels = np.stack(one_hot_labels, axis=0)
    return one_hot_labels.astype(np.int)


def get_labels_from_one_hot(preds):
    labels = []
    for pred in preds:
        labels.append(np.where(np.array(pred) == 1)[0])
    return labels


def reverse_mapping(predictions, dict_mapping):
    reversed_dict_mapping = {v: k for k, v in dict_mapping.items()}
    reversed_preds = []
    for pred in predictions:
        reversed_current = []
        for label in pred:
            reversed_current.append(reversed_dict_mapping[label])
        reversed_preds.append(reversed_current)
    return reversed_preds
