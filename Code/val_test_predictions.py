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
from multiprocessing_utils import *
from IPython import embed

N_JOBS = 15
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
        dct = clean_and_process_file(dataset, fname)
        if dct == '<ERROR>':
            continue
        article = dct['full_text']
        sentences.extend(article)
    return sentences

def clean_files(dataset, file_list, JobQueue):
    np.random.seed()
    sentences_predict_proba = {}
    sentences = get_sentences(dataset, file_list)
    rand_fname = str(int(10e10 * np.random.rand()) )
    tmp_sent_file = "../TMP/" + rand_fname + ".txt"
    tmp_save_file = "../TMP/" + rand_fname + ".pkl"

    #get embeddings
    write_file = open(tmp_sent_file, "w")

    #To avoid memory problems
    n = len(sentences)
    i = 0
    h = 1000
    while i < n:
        s = i
        e = min(n, i + h)
        write_file.write(u"\n".join(sentences[s:e]).encode("utf-8"))
        i = i + h
    write_file.close()

    cmd = ["python", "get_sentence_embeddings.py", "--sentences_file", tmp_sent_file, "--save_file", tmp_save_file, "--cpu"]
    subprocess.call(cmd)

    embeddings_dct = pickle.load(open(tmp_save_file, "rb"))
    JobQueue.put(embeddings_dct)

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

#multiprocessing functions
def processor(file_list, dataset,  JobQueue):
    sentences = get_sentences(dataset, file_list)
    JobQueue.put(sentences)

def tmpwriter(JobQueue):
    sentences = []
    while True:
        res = JobQueue.get()
        if res == "kill":
            break
        sentences.extend(res)
    return sentences

#model =(model, mu, sig)
def predictions_processor(sentences, model, embeddings_dct, JobQueue):
    embeddings = []

    print("Inside predictor")
    for sent in tqdm(sentences):
        try:
            embeddings.append(embeddings_dct[sent])
        except:
            print("No embedding found for <" + sent + "> Setting probability to 1.0")
            embeddings.append(np.random.rand(512))

    mu = model[1]
    sig = model[2]
    model = model[0]
    x = normalize(embeddings, mu, sig)

    prob = []
    h = 100
    for i in tqdm(range(0, len(sentences), h)):
        prob.extend(model.predict_proba(x[i:i+h]))

    prob_dct = {}
    for i in range(0, len(sentences)):
        sent = sentences[i]
        prob_dct[sent] = prob[i]

    JobQueue.put(prob_dct)

def predictions_writer(save_file, JobQueue):
    print("inside predictor writer")
    prob_dct = {}
    while True:
        res = JobQueue.get()
        if res == "kill":
            break

        for sent in res:
            prob_dct[sent] = res[sent]

    pickle.dump(prob_dct, open(save_file, "wb"))
    return prob_dct

def convert_dct_to_mp_sharing(dct, manager, delete_orig = True):
    manager_dct = manager.dict()
    keys = list(dct.keys())

    for key in tqdm(keys):
        manager_dct[key] = dct[key]
        if delete_orig:
            del dct[key]
    return manager_dct

def main(dataset, type_s, parallelism = 4, force_create_embeddings = False, force_new_predictions = False):
    file_list = open(os.path.join("../Data/Processed_Data/", dataset, "All/test_file_list.txt")).read().strip().split("\n")
    file_list.extend(open(os.path.join("../Data/Processed_Data/", dataset, "All/val_file_list.txt")).read().strip().split("\n"))

    save_file = os.path.join("../Data/Processed_Data", dataset,  "pkl_files", "test_val_embeddings.pkl")

    if not os.path.exists ("../TMP"):
        os.mkdir("../TMP")

    if force_create_embeddings or not os.path.exists(save_file):
        n = len(file_list)
        h = 1 + int(n/parallelism)

        pool = mp.Pool()
        manager = mp.Manager()
        JobQueue = manager.Queue()
        jobs = []
            
        writer = pool.apply_async(embeddings_writer, (save_file, JobQueue, ))

        for i in range(0, n, h):
            job = pool.apply_async(clean_files, (dataset, file_list[i:i+h], JobQueue))
            jobs.append(job)

        for job in jobs:
            job.get()

        JobQueue.put("kill")
        writer.get()
        pool.close()
        pool.join()

    manager = mp.Manager()
    pool = mp.Pool()
    jobs = []
    embeddings_file = save_file
    embeddings_dct = None

    if embeddings_dct == None:
        print("Loading Sentence Embeddings - This while takes some time")
        embeddings_dct = pickle.load(open(embeddings_file, "rb"))
        print("Done")
        print("Converting embeddings_dct to Manager dict and deleting embeddings_dct")
        embeddings_dct = convert_dct_to_mp_sharing(embeddings_dct, manager, delete_orig=True)
        print("Done")

    for domain in DOMAINS:
        save_file = os.path.join("../Data/Processed_Data/", dataset, domain, type_s, "predictions.pkl")

        test_file = os.path.join("../Data/Processed_Data/", dataset, domain, "test_file_list.txt")
        file_list = open(test_file).read().strip().split("\n")
        val_file = os.path.join("../Data/Processed_Data/", dataset, domain, "val_file_list.txt")
        file_list.extend(open(val_file).read().strip().split("\n"))

        ################ sentences = get_sentences(dataset, file_list)
        print("Getting sentences: " + str(len(file_list)) + " files")
        sentences = multiprocessing_func(processor, file_list, 15, [dataset], tmpwriter, [])
        print("Done")

        if force_new_predictions or not os.path.exists(save_file):
            print("Getting model for " + dataset + " - " + domain)
            modelfile = os.path.join("../Data/Processed_Data", dataset, domain, type_s, "model.pkl")
            model = pickle.load(open(modelfile, "rb"))
            print("Done")
            print("Predicting ...")
            sentence_predictions_dct = multiprocessing_func(predictions_processor, sentences, 15, (model, embeddings_dct), predictions_writer, [save_file])
            print("Done")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default=None)
    args.add_argument("-type_s", type=str, default="importance")
    args.add_argument("-parallelism", type=int , default=20)
    args.add_argument("--force_create_embeddings", action="store_true")
    args.add_argument("--force_new_predictions", action="store_true")
    opts = args.parse_args()

    if opts.dataset not in ["nyt"]:
        DOMAINS = ["All"]
    opts = args.parse_args()
    main(opts.dataset, opts.type_s, opts.parallelism, opts.force_create_embeddings, opts.force_new_predictions)
