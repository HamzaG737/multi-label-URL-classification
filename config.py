import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.set_defaults(eval=True)

    parser.add_argument("--preprocess", dest="preprocess", action="store_true")
    parser.set_defaults(preprocess=False)

    # data args
    parser.add_argument(
        "--path_zip_data",
        default="data/",
        help="path to zip dataset",
    )

    parser.add_argument(
        "--path_stopwords_list",
        default="data/stop_words_french.txt",
        help="path to a list of stopwords used for preprocessing",
    )

    parser.add_argument(
        "--path_test_data",
        default="data/df_test.csv",
        help="path to test data ",
    )

    parser.add_argument(
        "--path_train_data",
        default="data/df_train.csv",
        help="path to train data ",
    )

    parser.add_argument(
        "--preprocessing_class_path",
        default="data/preprocess_instance.pkl",
        help="path to preprocessing instance ",
    )

    parser.add_argument(
        "--path_results",
        default="data/results.csv",
        help="path to store the url's categories.",
    )

    # fasttext models args
    parser.add_argument(
        "--training_data_fastext_path",
        default="data/training_data_fasttext.txt",
        help="path to zip dataset",
    )

    parser.add_argument(
        "--fast_text_path",
        default="models/fast_text_model.bin",
        help="path to zip dataset",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=2 / 3,
        help="weight reduce in fasttext embeddings compute",
    )
    # classifier models args
    parser.add_argument(
        "--classifier_name",
        default="mlaram",
        choices=["mlaram", "mlknn"],
        help="name of the classifier for the multi-label classification",
    )

    parser.add_argument(
        "--mlknn_k",
        type=int,
        default=3,
        help="number of neighbours for the mlknn classifier",
    )

    parser.add_argument(
        "--vigilance",
        type=float,
        default=0.95,
        help="vigilance parameter for the mlaram classifer",
    )
    parser.add_argument(
        "--thresh_mlaram",
        type=float,
        default=5 * 1e-5,
        help="threshold param for mlaram classifier",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/classifier.pkl",
        help="path to save the classifier",
    )
    return parser