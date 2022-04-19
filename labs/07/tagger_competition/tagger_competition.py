#!/usr/bin/env python3
#
# Team members' IDs:
# 182a3da8-8a9e-11ec-986f-f39926f24a9c  (Jan Zubáč)
# 7797f596-9326-11ec-986f-f39926f24a9c
# 449dba85-9adb-11ec-986f-f39926f24a9c
#
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

import tagger_cle

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
# parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
# parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
# parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=6, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")

parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.0, type=float, help="Mask words with the given probability.")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Create the model and train it
    model =  tagger_cle.Model(args, morpho.train)

    def extract_tagging_data(example):
        input  = example["forms"]
        tag_ids = morpho.train.tags.word_mapping(example["tags"])    
        return input, tag_ids

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset

    train, dev = create_dataset("train"), create_dataset("dev")
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set; update the following prediction
        # command if you use other output structre than in tagger_we.
        test = create_dataset("test")
        predictions = model.predict(test)
        tag_strings = morpho.test.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in sentence:
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
