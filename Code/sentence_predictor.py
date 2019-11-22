import os
import pickle
import argparse
import nltk
# from val_test_predictions import normalize
from copy import deepcopy
from extract_documents import *
from tqdm import tqdm
import numpy as np

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations", "All"]

def normalize(x, mu, sig):
    x = deepcopy(x)
    x = np.array(x)
    mu.resize(512, 1)
    sig.resize(512, 1)
    x = x.transpose()
    x =  (x - mu) / sig
    x = x.transpose()
    return x

def predict( embedddings_dct, model, threshold ):
    sentences = list(embedddings_dct.keys())
    pred = {}
    embeddings = []
    for sent in sentences:
        embeddings.append(embedddings_dct[sent])

    sklearn_model = model[0]
    mu = model[1]
    sig = model[2]

    embeddings = normalize(embeddings, mu, sig)
    prob = sklearn_model.predict_proba(embeddings)

    for i in range(0, len(sentences)):
        if prob[i][1] >= threshold:
            pred[sentences[i]] = 1
        else:
            pred[sentences[i]] = 0

    return pred

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-trained_on_dataset", type=str, default="nyt")
    # args.add_argument("-type_s", type=str, default=None) #Do not specify if you want both classifiers
    # args.add_argument("-split", type=str, default="valid") #valid/test
    # args.add_argument("-domain", type=str, default=None) #Business/Sports/Science/USIntlRelations/All=None
    # args.add_argument("-save_dir", type=str, default="../TMP/") #Business/Sports/Science/USIntlRelations/All=None
    args.add_argument("-threshold", type=float, default = 0.7) #Business/Sports/Science/USIntlRelations/All=None
    opts = args.parse_args()

    threshold = opts.threshold
    # if opts.split == "valid":
        # opts.split = "val"

    # if opts.domain == "None":
        # opts.domain = "All"
        
    model = pickle.load(open("../Data/Processed_Data/" + opts.trained_on_dataset + "/All/importance/model.pkl", "rb"))
    try:
        embedddings_dct = pickle.load(open("../Data/Processed_Data/Soham/pkl_files/shards/0/Sentence_embeddings.pkl", "rb"))
    except:
        embedddings_dct = pickle.load(open("../Data/Processed_Data/Soham/pkl_files/shards/0/Sentence_embeddings.pkl", "rb"), encoding="latin1")
    save_file = "../Data/Processed_Data/Soham/pkl_files/shards/0/" + opts.trained_on_dataset + "." + str(threshold) + ".predictions.pkl"
    pred = predict(embedddings_dct, model, threshold)
    pickle.dump(pred, open(save_file, "wb"))

