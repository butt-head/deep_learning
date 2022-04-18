#!/usr/bin/env python3
#
# Team members' IDs:
# 182a3da8-8a9e-11ec-986f-f39926f24a9c  (Jan Zubáč)
# 7797f596-9326-11ec-986f-f39926f24a9c
# 449dba85-9adb-11ec-986f-f39926f24a9c
import argparse
import datetime
import os
import re
from typing import Any, Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.0, type=float, help="Mask words with the given probability.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    # A layer setting given rate of elements to zero.
    class MaskElements(tf.keras.layers.Layer):
        def __init__(self, rate: float) -> None:
            super().__init__()
            self._rate = rate

        def get_config(self) -> Dict[str, Any]:
            return {"rate": self._rate}

        def call(self, inputs: tf.RaggedTensor, training: bool) -> tf.RaggedTensor:
            if training:
                # TODO: Generate as many random uniform numbers in range [0, 1) as there are
                # values in `tf.RaggedTensor` `inputs` using a single `tf.random.uniform` call
                # (without setting seed in any way, so with just a single parameter `shape`).
                # Then, set the values in `inputs` to zero if the corresponding generated
                # random number is less than `self._rate`.
                # raise NotImplementedError()

                rand_nums = tf.random.uniform(tf.shape(inputs.flat_values))
                print("rand_nums:", rand_nums)
                # mask = tf.constant(rand_nums>=self._rate)
                mask = rand_nums>=self._rate
                # print("mask:", mask)
                # print("inputs:", inputs)
                mask = tf.cast(mask, dtype=tf.int64)

                inputs = inputs*mask
            else:
                return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        # Implement a one-layer RNN network. The input `words` is
        # a `RaggedTensor` of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        words_indices = train.forms.word_mapping(words)
        vocab_size = train.forms.word_mapping.vocabulary_size()

        # TODO: With a probability of `args.word_masking`, replace the input word by an
        # unknown word (which has index 0).
        #
        # There are two approaches you can use:
        # 1) use the above defined `MaskElements` layer, in which you need to implement
        #    one TODO note. If you do not want to implement it, you can instead
        # 2) use a `tf.keras.layers.Dropout` to achieve this, even if it is a bit
        #    hacky, because Dropout cannot process integral inputs. Start by using
        #    `tf.ones_like` to create a ragged tensor of float32 ones with the same
        #    structure as the indices of the input words, pass them through a dropout layer
        #    with `args.word_masking` rate, and finally set the input word ids to 0 where
        #    the result of dropout is zero.

        ## 1)
        # words_indices_replaced = self.MaskElements(args.word_masking) (words_indices)
        ## 2)
        mask = tf.keras.layers.Dropout(args.word_masking)(tf.ones_like(words_indices, dtype=tf.float32))
        words_indices_replaced = tf.where(mask > 0, words_indices, tf.cast(0, dtype=tf.int64)) # if mask > 0 then use original else make 0
        
        # # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        words_embedded = tf.keras.layers.Embedding(vocab_size, args.we_dim)(words_indices_replaced)

        # # TODO: Create a vector of input words from all batches using `words.values`
        # # and pass it through `tf.unique`, obtaining a list of unique words and
        # # indices of the original flattened words in the unique word list.

        vec_unique_words, ind_unique_words  = tf.unique(words.values)  # !!! words_embedded.values
        # ## vec_unique_words, ind_unique_words  = tf.unique(words_embedded.values)  # !!! words_embedded.values


        # # TODO: Create sequences of letters by passing the unique words through
        # # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        seqs_letters = tf.strings.unicode_split(vec_unique_words, "UTF-8")

        # # TODO: Map the letters into ids by using `char_mapping` of `train.forms`.
        letters_ids = train.forms.char_mapping(seqs_letters)
        input_size_embedded = train.forms.char_mapping.vocabulary_size()

        # # TODO: Embed the input characters with dimensionality `args.cle_dim`.
        letters_embedded = tf.keras.layers.Embedding(input_size_embedded, args.cle_dim)(letters_ids)

        # # TODO: Pass the embedded letters through a bidirectional GRU layer
        # # with dimensionality `args.cle_dim`, obtaining character-level representations
        # of the whole words, **concatenating** the outputs of the forward and backward RNNs.
        
        # rnn_layer = tf.keras.layers.GRU(args.cle_dim, return_sequences=True) ## !!!!
        rnn_layer = tf.keras.layers.GRU(args.cle_dim) ## !!!!

        out_rnn = tf.keras.layers.Bidirectional(rnn_layer, merge_mode='concat')(letters_embedded)

        # # TODO: Use `tf.gather` with the indices generated by `tf.unique` to transform
        # # the computed character-level representations of the unique words to representations
        # # of the flattened (non-unique) words.
        words_char_lev_flattened = tf.gather(out_rnn, ind_unique_words)

        # # TODO: Then, convert these character-level word representations into
        # # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        words_char_lev_ragged = words.with_values(words_char_lev_flattened)


        # # TODO: Concatenate the word-level embeddings and the computed character-level WEs
        # # (in this order).
        words_chars_embedded = tf.keras.layers.concatenate([words_embedded, words_char_lev_ragged])
        # words_chars_embedded = tf.keras.layers.concatenate([words_embedded, words_char_lev_ragged])
        # words_chars_embedded = words_embedded # !!!!

        # TODO(tagger_we): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.

        if args.rnn_cell == 'LSTM':
            rnn_layer = tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True)
        if args.rnn_cell == 'GRU':
            rnn_layer = tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True)

        out = tf.keras.layers.Bidirectional(rnn_layer, merge_mode='sum')(words_chars_embedded)

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        n_classes   = train.tags.word_mapping.vocabulary_size()  
        predictions = tf.keras.layers.Dense(n_classes, activation='softmax')(out)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=lambda yt, yp: tf.losses.SparseCategoricalCrossentropy()(yt.values, yp.values),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> Dict[str, float]:
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

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # TODO(tagger_we): Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integral tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.

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

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
