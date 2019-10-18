import os
import pickle
import multiprocessing as mp
import argparse
import numpy as np
from copy import deepcopy
from multiprocessing_utils import *

def normalize(x, mu, sig):
    x = deepcopy(x)
    x = np.array(x)
    mu.resize(512, 1)
    sig.resize(512, 1)
    x = x.transpose()
    x =  (x - mu) / sig
    x = x.transpose()
    return x


def load_embeddings(dataset):
    pkl_file = "../Data/Processed_Data/" + dataset + "/pkl_files/test_val_embeddings.pkl"
    try:
        embeddings = pickle.load(open(pkl_file, "rb"))
    except:
        embeddings = pickle.load(open(pkl_file, "rb"), encoding="latin1")

    embeddings_list = []
    cnt = 1
    for sent in embeddings:
        embeddings_list.append((sent, embeddings[sent]))
        if cnt > 100:
            break
    return embeddings_list

def load_classifier(classifier_file):
    model = pickle.load(open(classifier_file, "rb"))
    return model

def writer(save_file, JobQueue):
    dct = {}
    while True:
        res = JobQueue.get()
        if res == "kill":
            break

        embeddings_dct = res
        for sent in embeddings_dct:
            dct[sent] = embeddings_dct[sent]
    print("Saving Files ...")
    pickle.dump(dct, open(save_file, "wb"))
    print("Done")

def predict( embeddings_list, classifier, JobQueue):
    predictions = {}
    sentences, embeddings = zip(* embeddings_list)
    
    #normalize embeddings
    embeddings = np.array(embeddings)
    mu = classifier[1]
    sig = classifier[2]
    model = classifier[0]
    embeddings = normalize(embeddings, mu, sig)

    predictions_list = model.predict_proba(embeddings)

    for i in range(0, len(sentences)):
        predictions[sentences[i]] = predictions_list[i]

    JobQueue.put(predictions)

def main(opts):
    ################################
    #Load data, classifier
    ################################
    print("Loading Embeddings and Classifier ...")
    classifier_file = os.path.join("../Data/Processed_Data/",
                                    opts.classifier_dataset,
                                    opts.classifier_domain,
                                    opts.type_s,
                                    "model.pkl"
                                    )
    classifier = load_classifier(classifier_file)
    embeddings_list = load_embeddings(opts.dataset)

    ################################
    #Make relevant directories
    ################################
    print("Creating directories")
    if not os.path.exists("../Data/Cross_Classifier_Predictions"):
        os.mkdir("../Data/Cross_Classifier_Predictions")
    save_dir = os.path.join("../Data/Cross_Classifier_Predictions/", 
                            opts.classifier_dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_file = os.path.join(save_dir, opts.dataset + ".pkl")
    f = open(save_file, "w")
    f.write("hello")
    f.close()

    ################################
    #Paralley predict
    ################################
    print("Parallely predicting .. ")
    multiprocessing_func(predict, embeddings_list, 24, 
                            [classifier], writer, [save_file])

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-classifier_dataset", type=str, required=True)
    args.add_argument("-classifier_domain", type=str, default="All")
    args.add_argument("-dataset", type=str, required=True)
    args.add_argument("-type_s", type=str, default="importance")
    opts = args.parse_args()
    main(opts)
