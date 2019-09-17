from __future__ import print_function
import os
import sys
import pickle
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations"]

pos_UPPER_THRESHOLD = 0.8
pos_LOWER_THRESHOLD = 0.6

neg_UPPER_THRESHOLD = 0.5
neg_LOWER_THRESHOLD = 0.5

#Edit as per requirement
UPPER_THRESHOLD = -1
LOWER_THRESHOLD = -1
#

TRAIN_TEST_SPLIT = 0.6
MAX_ARTICLES = 40000 # maximum number of articles in each class (pos/neg). -1 for no limit

def create_training_data(domain_save_dir, shard_dir, subdir, domain, JobQueue):
    load_file = os.path.join(shard_dir, subdir, "Sentence_pairs_scores.pkl" )
    Sentence_pairs = pickle.load(open(load_file, "rb"))
    load_file = os.path.join(shard_dir, subdir, "Sentence_embeddings.pkl" )
    embedding_dct = pickle.load(open(load_file, "rb"))

    embedding = []
    label = []
    cos = []

    f = open(os.path.join(domain_save_dir, "train_file_list.txt"))
    domain_train_file_list = set(f.read().strip().split("\n"))

    for sent in tqdm(Sentence_pairs):
        s = Sentence_pairs[sent]

        if s['filename'] not in domain_train_file_list:
            continue

        #This excludes all summary sentences
        if "pairs" not in s:
            continue

        if s['max_score'] >= UPPER_THRESHOLD or s['max_score'] <= LOWER_THRESHOLD:
            embedding.append(embedding_dct[sent])
            cos.append(s["max_score"])

            if s['max_score'] >= UPPER_THRESHOLD:
                label.append(1)
            else:
                label.append(-1)

    label = np.array(label)
    embedding = np.array(embedding)
    cos = np.array(cos)

    if MAX_ARTICLES != -1:
        rand = np.random.permutation(len(label))[: MAX_ARTICLES]
        label = label[rand]
        embedding = embedding[rand]
        cos = cos[rand]

    dct = {
            "embedding": embedding,
            "label": label,
            "cos": cos,
            }
    JobQueue.put(dct)

def merge_data(JobQueue, type_s, save_dir):
    dct = {
            "embedding": [],
            "label": [],
            "cos": [],
            }

    while True:
        res = JobQueue.get()
        if res == "kill":
            train_test_split(dct, save_dir, type_s)
            return

        dct['embedding'].extend(res['embedding'])
        dct['label'].extend(res['label'])
        dct['cos'].extend(res['cos'])

def train_test_split(dct, save_dir, type_s):
    print("Splitting Data")
    global NUM_TEST_SAMPLES_pos, NUM_TEST_SAMPLES_neg
    global NUM_TRAIN_SAMPLES_pos, NUM_TRAIN_SAMPLES_neg
    global MAX_ARTICLES

    label = np.array(dct['label'])
    embedding = np.array(dct['embedding'])
    cos = np.array(dct['cos'])

    pos_label_indexes = [i for i in range(0, len(label)) if label[i] == 1]
    neg_label_indexes = [i for i in range(0, len(label)) if label[i] == -1]

    print("Number of positive labels = " + str(len(pos_label_indexes)))
    rpos = np.random.permutation(pos_label_indexes)
    rneg = np.random.permutation(neg_label_indexes)

    npos = len(pos_label_indexes)
    nneg = len(neg_label_indexes)
    n = min(npos, nneg)

    NUM_TRAIN_SAMPLES_pos = int(TRAIN_TEST_SPLIT * n)
    NUM_TRAIN_SAMPLES_neg = int(TRAIN_TEST_SPLIT * n)
    NUM_TEST_SAMPLES_pos = int((1-TRAIN_TEST_SPLIT) * n)
    NUM_TEST_SAMPLES_neg = int((1-TRAIN_TEST_SPLIT) * n)

    if MAX_ARTICLES != -1:
        NUM_TRAIN_SAMPLES_pos = min(NUM_TRAIN_SAMPLES_pos, MAX_ARTICLES)
        NUM_TRAIN_SAMPLES_neg = min(NUM_TRAIN_SAMPLES_neg, MAX_ARTICLES)
        NUM_TEST_SAMPLES_pos = min(NUM_TEST_SAMPLES_pos, MAX_ARTICLES)
        NUM_TEST_SAMPLES_neg = min(NUM_TEST_SAMPLES_neg, MAX_ARTICLES)

    train = {
            "embedding": [],
            "label": [],
            "cos": []
            }

    train['embedding'].extend(embedding[rpos][:NUM_TRAIN_SAMPLES_pos])
    train['label'].extend(label[rpos][:NUM_TRAIN_SAMPLES_pos])
    train['cos'].extend(cos[rpos][:NUM_TRAIN_SAMPLES_pos])

    train['embedding'].extend(embedding[rneg][:NUM_TRAIN_SAMPLES_neg])
    train['label'].extend(label[rneg][:NUM_TRAIN_SAMPLES_neg])
    train['cos'].extend(cos[rneg][:NUM_TRAIN_SAMPLES_neg])

    test = {
            "embedding": [],
            "label": [],
            "cos": [],
            }
    test['embedding'].extend(embedding[rpos][-NUM_TEST_SAMPLES_pos:])
    test['label'].extend(label[rpos][-NUM_TEST_SAMPLES_pos:])
    test['cos'].extend(cos[rpos][-NUM_TEST_SAMPLES_pos:])

    test['embedding'].extend(embedding[rneg][-NUM_TEST_SAMPLES_neg:])
    test['label'].extend(label[rneg][-NUM_TEST_SAMPLES_neg:])
    test['cos'].extend(cos[rneg][-NUM_TEST_SAMPLES_neg:])

    print("Positive train samples: {}\nNegative train samples: {}".format(str(NUM_TRAIN_SAMPLES_pos), str(NUM_TRAIN_SAMPLES_neg) ) )
    print("Positive test samples: {}\nNegative test samples: {}".format(str(NUM_TEST_SAMPLES_pos), str(NUM_TEST_SAMPLES_neg) ) )

    #Save files
    print("Saving data to ", end = "")
    save_file = os.path.join(save_dir, type_s, "train_data.pkl")
    print(save_file, end =" and ")
    pickle.dump(train, open(save_file, "wb"))
    save_file = os.path.join(save_dir, type_s, "test_data.pkl")
    print(save_file)
    pickle.dump(test, open(save_file, "wb"))

#type_s = importance/ throwaway
def create_train_test_data(dataset, type_s, domain = "All"):
    save_dir = os.path.join("../Data/Processed_Data", dataset, domain)
    shard_dir = os.path.join("../Data/Processed_Data", dataset, "pkl_files/shards")

    if not os.path.exists(os.path.join(save_dir, type_s)):
        os.mkdir(os.path.join(save_dir, type_s))

    global UPPER_THRESHOLD, LOWER_THRESHOLD
    global pos_UPPER_THRESHOLD, pos_LOWER_THRESHOLD
    global neg_UPPER_THRESHOLD, neg_LOWER_THRESHOLD

    if type_s == 'importance':
        UPPER_THRESHOLD = pos_UPPER_THRESHOLD
        LOWER_THRESHOLD = pos_LOWER_THRESHOLD
    else:
        UPPER_THRESHOLD = neg_UPPER_THRESHOLD
        LOWER_THRESHOLD = neg_LOWER_THRESHOLD

    print("type_s = " + type_s, UPPER_THRESHOLD, LOWER_THRESHOLD)

    manager = mp.Manager()
    JobQueue = manager.Queue()
    pool = mp.Pool()
    jobs = []

    merger = pool.apply_async(merge_data, (JobQueue, type_s, save_dir))
    for subdir in os.listdir(shard_dir):
        #ensure only training shards are used
        if not subdir.startswith("train"):
            continue

        job = pool.apply_async(create_training_data, (save_dir, shard_dir, subdir, domain, JobQueue))
        jobs.append(job)

    for job in jobs:
        job.get()

    JobQueue.put("kill")
    merger.get()
    pool.close()
    pool.join()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default="nyt")
    args.add_argument("-type_s", type=str, default="importance")
    opts = args.parse_args()

    create_train_test_data(opts.dataset, opts.type_s)

    if opts.dataset in ["cnn", "cnndm", "gigaword"]:
        DOMAINS = []

    for domain in DOMAINS:
        create_train_test_data(opts.dataset, opts.type_s, domain)
