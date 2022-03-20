#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The neural network model
class Model(tf.keras.Model):



    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in the variable `hidden`.
        # hidden = ...

        def get_layers_parameters(args_str):
            layers, parameters = [], []
            
            i_R, i_ = None, None
            splitted = [arg for arg in re.split(r",", args_str)    ]
            
            for i, _ in enumerate(splitted):
                if splitted[i][0]=='R':
                    i_R = i
                if splitted[i][-1].endswith(']'):
                    i_ = i
            
            splitted_new = [] 
            if i_R != None:
                i = 0
                splitted_R = ''
                while i < len(splitted):
                    if (i < i_R) or (i > i_):
                        splitted_new.append(splitted[i]) 
                    else: 
                        while i <i_:
                            splitted_R = splitted_R +','+ splitted[i]
                            i += 1
                        splitted_R = splitted_R +','+ splitted[i]
                        splitted_new.append(splitted_R[1:])
                    i += 1
            else:
                splitted_new = splitted

            for args_ in splitted_new:
                if args_[0] == 'R':
                    layers.append(args_[0])
                    parameters.append(args_[2:])
                else:
                    arg_list = args_.split('-') 
                    layers.append(arg_list[0])
                    parameters.append(arg_list[1:])

            return layers, parameters

        def convolution(inputs, parameters):
                filters, kernel_size, stride, padding = parameters
                filters, kernel_size, stride = int(filters), int(kernel_size), int(stride)

                outputs = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding,
                    data_format=None, dilation_rate=(1, 1), groups=1, activation='relu',
                    use_bias=True, kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', kernel_regularizer=None,
                    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                    bias_constraint=None)(inputs)
                return outputs

        def convolution_batch_norm(inputs, parameters):
            filters, kernel_size, stride, padding = parameters
            # print("padding:",padding)
            filters, kernel_size, stride = int(filters), int(kernel_size), int(stride)

            outputs = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding,
                data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
                use_bias=False, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None)(inputs)

            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.ReLU()(outputs)
            return outputs



        layers, parameters = get_layers_parameters(args.cnn)
        outputs = inputs     

        for i, layer in enumerate(layers):
            if layer == 'C':   # convolution
                outputs = convolution(outputs, parameters[i])

            if layer == 'CB':  # convolution with batch norm
                outputs = convolution_batch_norm(outputs, parameters[i])

            if layer == 'M': # max pooling
                pool_size, stride = [int(p) for p in parameters[i]]
                outputs = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=stride)(outputs)

            if layer == 'R':  # residual connection
                inputs_R = outputs
                layers_res, parameters_res = get_layers_parameters(parameters[i][ parameters[i].index("[")+1 : parameters[i].index("]") ] )
        
                print(parameters[i][ parameters[i].index("[")+1 : parameters[i].index("]") ] )
                print(layers_res)
                print(parameters_res)
                for i, layer_res in enumerate(layers_res):
                    if layer_res == 'C':   # convolution
                        outputs = convolution(outputs, parameters_res[i])
                    if layer_res == 'CB':  # convolution with batch norm
                        outputs = convolution_batch_norm(outputs, parameters_res[i]) 
                outputs = inputs_R + outputs 

            if layer == 'F':  # flatten
                outputs = tf.keras.layers.Flatten() (outputs)

            if layer == 'H':  # hidden
                hidden_layer_size = int(*parameters[layers.index('H')])
                outputs = tf.keras.layers.Dense(hidden_layer_size, activation='relu') (outputs)

            if layer == 'D':  # dropout
                dropout_rate = float(*parameters[layers.index('D')])
                outputs = tf.keras.layers.Dropout(dropout_rate) (outputs)

        hidden = outputs

        # # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace): #-> Dict[str, float]:
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
    mnist = MNIST()

    # # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}
    # return


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
