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
from tqdm import tqdm
import subprocess

N_JOBS = 15
THRESHOLD = 0.6
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

def get_sentences(dataset, file_list):
    sentences = []
    for fname in tqdm(file_list):
        f = open(fname)
        dct = clean_and_process_file(dataset, fname)
        article = dct['full_text']
        # abstract = dct['abstract']
        sentences.extend(article)
    return sentences

def clean_files(dataset, file_list, threshold, JobQueue):
    np.random.seed()
    sentences_predict_proba = {}

    sentences = get_sentences(dataset, file_list)
    rand_fname = str(int(10e10 * np.random.rand()) )
    tmp_sent_file = "../TMP/" + rand_fname + ".txt"
    tmp_save_file = "../TMP/" + rand_fname + ".pkl"

    #get embeddings
    write_file = open(tmp_sent_file, "w")
    write_file.write("\n".join(sentences))
    write_file.close()

    cmd = ["python", "get_sentence_embeddings.py", "--sentences_file", tmp_sent_file, "--save_file", tmp_save_file, "--cpu"]
    subprocess.call(cmd)
    embeddings_dct = pickle.load(open(tmp_save_file, "rb"))
    JobQueue.put(embeddings_dct)

def sentence_predictions(dataset, file_list, model, embeddings_dct, save_file):
    sentences = get_sentences(dataset, file_list)
    #create np array from embeddings dct
    embeddings = []
    for sent in sentences:
        embeddings.append(embeddings_dct[sent])

    embeddings = np.array(embeddings)

    mu = model[1]
    sig = model[2]
    model = model[0]

    x = deepcopy(embeddings)
    x = normalize(x, mu, sig)
    prob = model.predict_proba(x)
    sentences_predict_proba = {}

    for i in range(0, len(sentences)):
        sent = sentences[i]
        sentences_predict_proba[sent] = prob[i]

    pickle.dump(sentences_predict_proba, open(save_file, "wb"))

def embeddings_writer(save_file, JobQueue):
    dct = {}
    while True:
        res = JobQueue.get()
        if res == "kill":
            break

        embeddings_dct = res
        for sent in embeddings_dct:
            dct[sent] = embeddings_dct[sent]
    pickle.dump(dct, open(save_file, "wb"))

def main(dataset, type_s, threshold, parallelism = 4, force_create_embeddings = False, force_new_predictions = False):
    if dataset == "nyt":
        file_list = open("../Data/Processed_Data/nyt/All/test_file_list.txt").read().strip().split("\n")
        file_list.extend(open("../Data/Processed_Data/nyt/All/val_file_list.txt").read().strip().split("\n"))

    save_file = os.path.join("../Data/Processed_Data", dataset,  "pkl_files", "test_val_embeddings.pkl")

    if force_create_embeddings or not os.path.exists(save_file):
        n = len(file_list)
        h = 1 + int(n/parallelism)

        pool = mp.Pool()
        manager = mp.Manager()
        JobQueue = manager.Queue()
        jobs = []
        writer = pool.apply_async(embeddings_writer, (save_file, JobQueue, ))

        dataset = "nyt"
        for i in range(0, n, h):
            job = pool.apply_async(clean_files, (dataset, file_list[i:i+h], threshold, JobQueue))
            jobs.append(job)

        for job in jobs:
            job.get()

        JobQueue.put("kill")
        writer.get()
        pool.close()
        pool.join()

    print("Loading Sentence Embeddings - This while takes some time")
    embeddings_dct = pickle.load(open(save_file, "rb"))
    print("Done")
    pool = mp.Pool()
    jobs = []

    for domain in DOMAINS:
        save_file = os.path.join("../Data/Processed_Data/", dataset, domain, type_s, "predictions.pkl")

        if force_new_predictions or not os.path.exists(save_file):
            file_list_file = os.path.join("../Data/Processed_Data/", dataset, domain, "test_file_list.txt")
            file_list = open(file_list_file).read().strip().split("\n")

            file_list_file = os.path.join("../Data/Processed_Data/", dataset, domain, "val_file_list.txt")
            file_list.extend( open(file_list_file).read().strip().split("\n") )


            modelfile = os.path.join("../Data/Processed_Data", dataset, domain, type_s, "model.pkl")
            model = pickle.load(open(modelfile, "rb"))
            job = pool.apply_async(sentence_predictions, (dataset, file_list, model, embeddings_dct, save_file))
            jobs.append(job)

    for job in jobs:
        job.get()
    pool.close()
    pool.join()



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default="nyt")
    args.add_argument("-type_s", type=str, default="importance")
    args.add_argument("-threshold", type=float, default=0.6)
    args.add_argument("-split_type", type=str , default="test")
    args.add_argument("-parallelism", type=int , default=20)
    args.add_argument("--force_create_embeddings", action="store_true")
    args.add_argument("--force_new_predictions", action="store_true")

    opts = args.parse_args()
    main(opts.dataset, opts.type_s, opts.threshold, opts.parallelism, opts.force_create_embeddings, opts.force_new_predictions)
