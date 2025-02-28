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
import functools

import numpy as np
import tensorflow as tf

from common_voice_cs import CommonVoiceCs

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# rnn pars
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=128, type=int, help="RNN cell dimension.")
# optimizers
parser.add_argument("--decay_steps", default=100, type=int, help="decay_steps.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.05, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=0.85, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
parser.add_argument("--decoder", default="greedy", type=str, help="Decoder to use( 'greedy' or 'beam_search' ).")


# parser.add_argument("--decoder", default="beam_search", type=str, help="Decoder to use( 'greedy' or 'beam_search' ).")
# parser.add_argument("--beam_search_width", default=10, type=int, help="width par used in beam search decoder")

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        inputs = tf.keras.layers.Input(shape=[None, CommonVoiceCs.MFCC_DIM], dtype=tf.float32, ragged=True)

        # TODO: Create a suitable model. You should:
        # - use a bidirectional RNN layer(s) to contextualize the input sequences.
        #
        # - optionally use suitable regularization
        #
        # - and finally generate logits for CTC loss/prediction as RaggedTensors.
        #   The logits should be generated by a dense layer with `1 + len(CommonVoiceCs.LETTERS)`
        #   outputs (the plus one is for the CTC blank symbol). Note that no
        #   activation should be used (the CTC operations will take care of it).

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True),
                                          merge_mode='sum')(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True),
                                          merge_mode='sum')(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True),
                                          merge_mode='sum')(x)
        out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True),
                                            merge_mode='sum')(x)

        n_classes = 1 + len(CommonVoiceCs.LETTERS)s
        logits = tf.keras.layers.Dense(n_classes, activation=None)(out)

        super().__init__(inputs=inputs, outputs=logits)

        # choosing optimizer
        if args.optimizer == 'SGD':
            if args.momentum != None:
                momentum = args.momentum
            else:
                momentum = 0.0
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False,
                                                name='SGD')
        if args.optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999,
                                                 epsilon=1e-07, amsgrad=False, name='Adam')

        # We compile the model with the CTC loss and EditDistance metric.
        # self._get_distribution_strategy = lambda: None   # my hack
        # self._distribution_strategy = self._get_distribution_strategy

        self.compile(optimizer=optimizer,
                     loss=self.ctc_loss,
                     metrics=[CommonVoiceCs.EditDistanceMetric()])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # TODO: Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the `gold_labels` to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tf.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.

        # gold_labels = tf.cast(gold_labels, 'int32')
        gold_labels_sparse = gold_labels.to_sparse(name="gold_labels_sparse")
        logits_dense = logits.to_tensor()
        # logits = tf.cast(logits, 'int32')
        # gold_labels = tf.cast(gold_labels, 'int32')

        logit_length = tf.cast(logits.row_lengths(), tf.int32)
        gold_labels_sparse = tf.cast(gold_labels_sparse, tf.int32)

        ctc_loss = tf.nn.ctc_loss(
            labels=gold_labels_sparse,
            logits=logits_dense,
            label_length=None,  # None,        #args.batch_size, #gold_labels.row_lengths(), #None, # None
            logit_length=logit_length,  # logit_length    logits_dense.row_lengths()
            logits_time_major=False,
            unique=None,
            blank_index=-1,
            name="ctc_loss"
        )

        return tf.math.reduce_mean(ctc_loss)
        # raise NotImplementedError()

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # TODO: Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor

        logits_dense = logits.to_tensor()
        logits_dense = tf.transpose(logits_dense, perm=[1, 0, 2])

        if args.decoder == 'greedy':
            predictions, _ = tf.nn.ctc_greedy_decoder(logits_dense,
                                                      tf.cast(logits.row_lengths(), tf.int32),
                                                      merge_repeated=True,
                                                      blank_index=-1
                                                      )

        if args.decoder == 'beam_search':
            predictions, _ = tf.nn.ctc_beam_search_decoder(logits_dense,
                                                           tf.cast(logits.row_lengths(), tf.int32),
                                                           beam_width=args.beam_search_width, top_paths=1)
        predictions = tf.RaggedTensor.from_sparse(predictions[0])

        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)


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

    # Load the data.
    cvcs = CommonVoiceCs()

    # # Create input data pipeline.
    def create_dataset(name):
        def prepare_example(example):
            # TODO: Create suitable batch examples.
            # - example["mfccs"] should be used as input
            # - the example["sentence"] is a UTF-8-encoded string with the target sentence
            #   - split it to unicode characters by using `tf.strings.unicode_split`
            #   - then pass it through the `cvcs.letters_mapping` layer to map
            #     the unicode characters to ids
            # raise NotImplementedError()
            input = example["mfccs"]
            output = tf.strings.unicode_split(example["sentence"], 'UTF-8')
            output = cvcs.letters_mapping(output)

            return (input, output)

        dataset = getattr(cvcs, name).map(prepare_example)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    # # TODO: Create the model and train it
    model = Model(args)
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS[char] for char in sentence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
