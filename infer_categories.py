from config import get_arguments
from load_data import LoadData
import pandas as pd
import pickle
from utils import get_labels_from_one_hot, reverse_mapping
from model import FastText


def process_input(config):
    """
    read the input using the paths in the config file.
    """
    with open(config.path_to_input, "r") as f:
        lines = f.readlines()
    urls = [line.replace("\n", "") for line in lines]
    return urls


if __name__ == "__main__":
    parser = get_arguments()
    parser.add_argument(
        "--path_to_input",
        type=str,
        required=True,
        help="path to input text file",
    )
    config = parser.parse_args()
    urls = process_input(config)
    load_instance = LoadData.load(config.preprocessing_class_path)
    processed_urls = [load_instance.parse_url(url) for url in urls]
    df_eval = pd.DataFrame(data={"text_url": processed_urls})
    fast_text = FastText(config, df_eval)
    X_test = fast_text.get_embeddings()

    # load classifer
    classifier = pickle.load(open(config.model_path, "rb"))
    print("Predicting ...")
    predictions = classifier.predict(X_test)
    labels = get_labels_from_one_hot(predictions)
    labels = reverse_mapping(labels, load_instance.dict_mapping)
    print("Saving results...")
    df_results = pd.DataFrame(data={"URL": urls, "targets": labels})
    df_results.to_csv(config.path_results, index=False)
    print(
        "Saving complete ! please check the {} path for the full results".format(
            config.path_results
        )
    )
