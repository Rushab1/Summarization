import os
import pickle
import argparse
import numpy as np
from copy import deepcopy
import tensorflow as tf
import tensorflow_hub as tf_hub
import multiprocessing as mp
from extract_documents import *
from gensim.summarization.summarizer import summarize

THRESHOLD = 0.6

def clean_files(file_list, modelfile, threshold):
    for fname in file_list:
        f = open(fname)
        try:
            f_pairs = json.load(f)
        except Exception as e:
            print("Cannot load json: " + fname + ": Error: " + str(e))
            continue

        for dct in f_pairs:
            source = dct['source']
            target = dct['target']
            local_sent.append(source)
            local_sent.append(target)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-type_s", type=str, default="by_sentence")
    args.add_argument("-threshold", type=float, default=0.6)
    args.add_argument("-test", type=int , default=1)
    args.add_argument("-valid", type=int, default=0)

    opts = args.parse_args()

    type_s = opts.type_s

    global THRESHOLD
    THRESHOLD = opts.threshold

    main1("./modelfiles/" + type_s + "/model.pkl", type_s, opts.test, opts.valid)
