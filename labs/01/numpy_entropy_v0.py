#!/usr/bin/env python3
import argparse
from collections import OrderedDict

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace): # -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    
    dic = {}
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            
            if line in dic:
                dic[line] += 1
            else:
                dic[line] = 1

              
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).


    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    p_distr = np.array(list(dic.values()))/sum(dic.values())

    # TODO: Load model distribution, each line `string \t probability`.
    dic2 = {}
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            key, val = line.split('\t')
            dic2[key] = val 
            # TODO: process the line, aggregating using Python data structures


    # TODO: Create a NumPy array containing the model distribution.
    q_distr = np.array(list(dic2.values()))

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = np.sum( -1* p_distr*np.log(p_distr))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.p

    # dict of tuples of pi and qi (handles the case when key of data is not present in keys of model)
    dic_both = dict()
    for i, (key, val) in enumerate(dic.items()):
        dic_both[key]=(p_distr[i], 0)
        

    for i, (key, val) in enumerate(dic2.items()):
        if key in dic_both.keys():
            dic_both[key]= (dic_both[key][0], q_distr[i])
        else:
            dic_both[key]= (0, q_distr[i])


    crossentropy = 0
    for key, val in dic_both.items():
        if val[1]==0:                # i. e. key is not present in the model
            crossentropy = np.inf
            break
        else:
            p, q = float(val[0]), float(val[1])
            crossentropy -= p*np.log(q)



    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
