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
def normalize(x, mu, sig):
    x = deepcopy(X)
    x = np.array(x)
    mu.resize(512, 1)
    sig.resize(512, 1)
    x = x.transpose()
    x =  (x - mu) / sig
    x = x.transpose()
    return x

def clean_files(dataset, file_list, model, threshold, JobQueue):
    sentences = []
    sentences_predict_proba = {}
    mu = model[1]
    sig = model[2]
    model = model[0]

    try:
        for fname in tqdm(file_list):
            f = open(fname)
            dct = clean_and_process_file(dataset, fname)

            article = dct['full_text']
            abstract = dct['abstract']

            sentences.extend(article)

        tmp_sent_file = "../TMP/" + str(np.random.rand())
        tmp_save_file = "../TMP/" + str(np.random.rand())
        open(tmp_sent_file, "w").write("\n".join(sentences))

        cmd = ["python", "get_sentence_embeddings.py", "--sentences_file", tmp_sent_file, "--save_file", tmp_save_file, "--cpu"]
        subprocess.call(cmd)
        embedddings = pickle.load(open(tmp_save_file, "rb"))

        prob = model.predict_proba(embeddings)

        for i in range(0, len(sentences)):
            sent = sentences[i]
            sentences_predict_proba[sent] = prob[i]

        JobQueue.put(sentences_predict_proba)

    except Exception as e:
        print(e)
        return

def predictions_writer(save_file, JobQueue):
    dct = {}
    while True:
        res = JobQueue.get()
        if res == "kill":
            break

        sentences_predict_proba = res
        for sent in sentences_predict_proba:
            dct[key] = sentences_predict_proba[key]
    pickle.dump(dct, open(save_file))


def main(dataset, domain, type_s, threshold, split_type="test", parallelism = 4):
    if split_type == "valid":
        split_type = "val"

    if dataset == "nyt":
        file_list = open("../Data/Processed_Data/nyt/" + domain + "/" + split_type + "_file_list.txt").read().strip().split("\n")
        modelfile = os.path.join("../Data/Processed_Data/nyt", domain, type_s, "model.pkl")
        model = pickle.load(open(modelfile, "rb"))
        save_file = os.path.join("../Data/Processed_Data/nyt", domain, type_s,  split_type + "_predictions.pkl")

    n = len(file_list)
    h = 1 + int(n/parallelism)

    pool = mp.Pool()
    manager = mp.Manager()
    JobQueue = manager.Queue()
    jobs = []
    writer = pool.apply_async(predictions_writer, (JobQueue, ))

    dataset = "nyt"
    for i in range(0, n, h):
        job = pool.apply_async(clean_files, (dataset, file_list[i:i+h], model, threshold, JobQueue))
        jobs.append(job)

    for job in jobs:
        job.get()

    JobQueue.put("kill")
    counter.get()
    pool.close()
    pool.join()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default="nyt")
    args.add_argument("-type_s", type=str, default="importance")
    args.add_argument("-threshold", type=float, default=0.6)
    args.add_argument("-split_type", type=str , default="test")
    args.add_argument("-parallelism", type=int , default=15)

    opts = args.parse_args()
    main(opts.dataset, "All", opts.type_s, opts.threshold, opts.split_type, opts.parallelism)

