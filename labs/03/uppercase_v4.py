#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
# parser.add_argument("--alphabet_size", default=None, type=int, help="If given, use this many most frequent chars.")
# parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
# parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
# parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# parser.add_argument("--window", default=None, type=int, help="Window size to use.")
parser.add_argument("--alphabet_size", default=40, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=8, type=int, help="Window size to use.")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization.")
parser.add_argument("--hidden_layers", default=[100,200], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout regularization.")
parser.add_argument("--decay_steps", default=10000, type=int, help="decay_steps.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
# parser.add_argument("--learning_rate_final", default=0.005, type=float, help="Final learning rate.")
parser.add_argument("--learning_rate_final", default=0.05, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=0.85, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")



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

    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is represented by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32),
            tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
            tf.keras.layers.Flatten()])

    regularizer = tf.keras.regularizers.L2(l2=args.l2)
    for hidden_layer in args.hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu,
                                        kernel_regularizer=regularizer, bias_regularizer=None))
        # model.add(tf.keras.layers.Dropout(args.dropout, seed = args.seed))

    # print(uppercase_data.train.alphabet)
    # print(len(uppercase_data.train.alphabet))
    # model.add(tf.keras.layers.Dense(len(np.unique(uppercase_data.train.alphabet)), activation=tf.nn.softmax,
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax,
                                    kernel_regularizer=regularizer, bias_regularizer=None))

    model.summary()


    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(  args.learning_rate, args.decay_steps, 
                                                                    end_learning_rate=args.learning_rate_final, power=1.0,
                                                                    cycle=False, name='linear_lr_decay')


    if args.optimizer=='SGD':
        if args.momentum != None:
            momentum = args.momentum
        else:
            momentum = 0.0
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False, name='SGD')

    if args.optimizer=='Adam':
        optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')


    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
        )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
    model.fit(
            uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            callbacks=[tb_callback]
        )

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        y_pred = model.predict(uppercase_data.test.data["windows"])
        y_pred = np.argmax(y_pred, axis=1)
        print("y_pred:", np.unique(y_pred))

        text_str = uppercase_data.test.text
        text_upper = "".join(c.upper() if y_pred[i]==1 else c for i, c in enumerate(text_str))
        predictions_file.write(text_upper)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
