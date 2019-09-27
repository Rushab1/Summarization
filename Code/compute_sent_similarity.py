import argparse
import os, re, sys
import numpy as np
import json
import pickle
import random
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import hashlib
from IPython import embed
from copy import deepcopy
import multiprocessing as mp
import argparse

MAX_TRAIN_FILES_CNNDM = 100000
PER_TEST_FILES = 35.0
PER_VAL_FILES = 5.0
DOMAINS = ["Business", "Sports", "Science", "USIntlRelations"]

def get_file_lists_domain_cnndm(data_dir, domain):
    print(data_dir)
    test_file_list = [];
    val_file_list = [];
    file_list = [];

    for root, subdir, files in os.walk(os.path.join(data_dir, domain)):
        for filename in files:
            if filename.startswith("train"):
                file_list.append(os.path.join(root, filename))
            if filename.startswith("test"):
                test_file_list.append(os.path.join(root, filename))
            if filename.startswith("val"):
                val_file_list.append(os.path.join(root, filename))

    try:
        file_list = random.sample(file_list, MAX_TRAIN_FILES_CNNDM)
    except:
        pass
    return file_list, val_file_list, test_file_list

#Makes a file of all training and test file locations for every year given the news type
def get_file_lists_domain(data_dir, domain):
    if "cnndm" in data_dir or "gigaword" in data_dir:
        return get_file_lists_domain_cnndm(data_dir, domain)

    file_list = []
    test_file_list = []

    for root, subdir, files in os.walk(os.path.join(data_dir, domain)):
        for filename in files:
            file_list.append(os.path.join(root, filename))

    random.shuffle(file_list)

    num_files = len(file_list)
    num_test_files = int(PER_TEST_FILES/100.0 * num_files)
    num_val_files = int(PER_VAL_FILES/100.0 * num_files)

    test_file_list = file_list[-num_test_files:]
    file_list = file_list[ : (num_files - num_test_files) ]
    num_files = len(file_list)

    val_file_list = file_list[-num_val_files:]
    file_list = file_list[ : (num_files - num_val_files) ]
    num_files = len(file_list)

    print("{}\nTrain files: {}\n val Files: {}\nTest Files: {}\n".format(
                    domain, str(num_files), str(num_val_files), str(num_test_files) ))

    return file_list, val_file_list, test_file_list

def write_file_lists(data_dir, file_list, val_file_list, test_file_list):
    f = open(os.path.join(data_dir, "train_file_list.txt"), "w")
    f.write("\n".join(file_list))
    f.close()

    f = open(os.path.join(data_dir, "test_file_list.txt"), "w")
    f.write("\n".join(test_file_list))
    f.close()

    f = open(os.path.join(data_dir, "val_file_list.txt"), "w")
    f.write("\n".join(val_file_list))
    f.close()

#Some files may occur in more than one domains but we treat them as different files
def get_file_list(data_dir, dataset):
    global DOMAINS

    #Create necessary directories
    if not os.path.exists("../Data/Processed_Data/"):
        os.mkdir("../Data/Processed_Data/")

    if not os.path.exists("../Data/Processed_Data/" + dataset):
        os.mkdir("../Data/Processed_Data/" + dataset)

    all_file_list = []
    all_val_file_list = []
    all_test_file_list = []

    for domain in DOMAINS:
        file_list, val_file_list, test_file_list = get_file_lists_domain(data_dir, domain)

        all_file_list.extend(file_list)
        all_val_file_list.extend(val_file_list)
        all_test_file_list.extend(test_file_list)

        if not os.path.exists("../Data/Processed_Data/" + dataset + "/" + domain):
            os.mkdir("../Data/Processed_Data/" + dataset + "/" + domain)

        write_file_lists("../Data/Processed_Data/" + dataset + "/" + domain + "/",
                            file_list,
                            val_file_list,
                            test_file_list)
    domain = "All"
    if not os.path.exists("../Data/Processed_Data/" + dataset + "/" + domain):
        os.mkdir("../Data/Processed_Data/" + dataset + "/" + domain)

    write_file_lists("../Data/Processed_Data/" + dataset + "/" + domain + "/",
                        all_file_list,
                        all_val_file_list,
                        all_test_file_list)

    print("{}\nTrain files: {}\n val Files: {}\nTest Files: {}\n".format(
        domain, str(len(all_file_list)), str(len(all_val_file_list)), str(len(all_test_file_list)) ))

#parallely reading json files and extracting sentences
def json_sentence_extractor(file_list, shard_save_file, JobQueue):
    Sentences = []
    local_sent = []

    for fname in tqdm(file_list):
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

        local_sent = list(set(local_sent))
        Sentences.extend(local_sent)
        local_sent = []
    JobQueue.put((Sentences, shard_save_file))

def json_sentence_writer(JobQueue):
    while(True):
        res = JobQueue.get()
        if res == "kill":
            break
        sentences = res[0]
        shard_save_file = res[1]

        f = open(shard_save_file, "w")
        for i in range(0, len(sentences)):
            f.write(sentences[i] + "\n")
        f.close()

#Extracts all sentences from the dataset and dumps into a text file, this file is typically large ~1GB
#shard_size = number of files per shard
def get_sentences(dataset, shard_size):
    print("Extracting sentences")

    data_dir = os.path.join("../Data/Processed_Data/", dataset)
    save_dir = os.path.join(data_dir , "pkl_files")
    shard_dir = os.path.join(save_dir, "shards")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(shard_dir):
        os.mkdir(shard_dir)
    else:
        os.system("rm -rf " + os.path.join(shard_dir, "*"))

    #get list of all dataset files
    # for prefix in ["train", "val", "test"]:
    for prefix in ["train"]:
        print("Creating shards for " + prefix + " files")
        fname = os.path.join(data_dir, "All/" + prefix + "_file_list.txt")
        file_list = open(fname).read().strip().split("\n")
        create_shards(file_list, shard_size, save_dir, shard_dir, prefix)

    # file_list.extend(open(os.path.join(data_dir, "All/val_file_list.txt")).read().strip().split("\n") )
    # file_list.extend(open(os.path.join(data_dir, "All/test_file_list.txt")).read().strip().split("\n") )

def create_shards(file_list, shard_size, save_dir, shard_dir, prefix):
    #Set up multiprocessing
    #multiprocessing tools
    n = len(file_list)
    h = shard_size

    manager = mp.Manager()
    pool = mp.Pool()
    JobQueue = manager.Queue()
    jobs = []

    #multiprocessing begins
    json_writer = pool.apply_async(json_sentence_writer, (JobQueue, ))
    file_shard_map = {}
    shard_file_map = {}
    for i in range(0, 1 + int(1.0*n/h)):
        s = i*h; e = min((i+1) * h, n);

        local_shard_dir = os.path.join(shard_dir, prefix + "_shard_" + str(i))
        shard_save_file = os.path.join(local_shard_dir, "Sentences.txt")

        if not os.path.exists(local_shard_dir):
            os.mkdir(local_shard_dir)

        job = pool.apply_async(json_sentence_extractor, (file_list[s:e], shard_save_file, JobQueue))
        jobs.append(job)

        shard_file_map[i] = []
        for j in range(s, e):
            file_shard_map[file_list[j]] = i
            shard_file_map[i].append(file_list[j])

    for job in jobs:
        job.get()

    JobQueue.put("kill")
    json_writer.get()
    pool.close()
    pool.join()

    save_file = os.path.join(save_dir, prefix +"_file_shard_map.pkl")
    pickle.dump(file_shard_map, open(save_file, "wb"))

    save_file = os.path.join(save_dir, prefix +"_shard_file_map.pkl")
    pickle.dump(shard_file_map, open(save_file, "wb"))

def parallel_embedding_extractor(shard_dir, device):
    save_file = os.path.join(shard_dir, "Sentence_embeddings.pkl")
    sentences_file = os.path.join(shard_dir, "Sentences.txt")

    if device == "cpu":
        cmd = "python get_sentence_embeddings.py --sentences_file {} --save_file {} --{}".format(
                                                sentences_file, save_file, device)
        print(cmd)
    else:
        cmd = "python get_sentence_embeddings.py --sentences_file {} --save_file {} ".format(
                                                sentences_file, save_file)

    os.system(cmd)
    print("\n\nEMBEDDINGS Done: " + save_file)

def get_sentence_embeddings(dataset, parallelism, device):
    shard_dir = os.path.join("../Data/Processed_Data", dataset, "pkl_files/shards")
    shard_subdirs = os.listdir(shard_dir)

    n = len(shard_subdirs)
    h = 1 + int(1.0 * n/parallelism)

    i = 0
    while i < n:
        pool = mp.Pool()
        jobs = []
        for j in range(i, i + parallelism):
            if j >= n: break
            shard_subdir = os.path.join(shard_dir, shard_subdirs[j])
            job = pool.apply_async(parallel_embedding_extractor, (shard_subdir, device))
            jobs.append(job)

        for job in jobs:
            job.get()
        pool.close()
        pool.join()
        i += parallelism

#Cosine Similarity
def cos_similarity(s1, s2):
    sim = np.dot(s1, s2)
    sim /= np.linalg.norm(s1)
    sim /= np.linalg.norm(s2)
    return sim

def score_sentences_worker(shard_dir, subdir, shard_file_map):
        shard_num = int(subdir.split("_")[-1])
        file_list = shard_file_map[shard_num]
        Sentence_embeddings = pickle.load(open(os.path.join(shard_dir, subdir, "Sentence_embeddings.pkl"), "rb"))
        Sentence_pairs = {}

        for fname in tqdm(file_list):
            f = open(fname)

            try:
                f_pairs = json.load(f)
            except:
                print("Cannot load " + fname)
                continue

            for dct in f_pairs:
                source = dct['source']
                target = dct['target']

                try:
                    source_embed = Sentence_embeddings[source]
                    target_embed = Sentence_embeddings[target]
                except:
                    continue

                sim = cos_similarity(source_embed,target_embed)

                if source not in Sentence_pairs:
                    Sentence_pairs[source] = {}
                if target not in Sentence_pairs:
                    Sentence_pairs[target] = {}

                Sentence_pairs[source]['filename'] = fname
                Sentence_pairs[target]['filename'] = fname
                Sentence_pairs[source]['type'] = 'summary'
                Sentence_pairs[target]['type'] = 'article'

                if 'pairs' not in Sentence_pairs[target]:
                    Sentence_pairs[target]['pairs'] = {}
                    Sentence_pairs[target]['max_score'] = 0

                Sentence_pairs[target]['pairs'][source] = sim

                max_score = Sentence_pairs[target]['max_score']
                if sim > max_score:
                    Sentence_pairs[target]['max_score'] = sim

        #Save Files
        save_file = os.path.join(shard_dir, subdir, "Sentence_pairs_scores.pkl")
        pickle.dump(Sentence_pairs, open(save_file, "wb"))


################################################################
################################################################
#(cosine) Score pairs of sentences one from the article and the other from summary for each shard
def score_sentences(dataset, parallelism):
    save_dir =  os.path.join("../Data/Processed_Data", dataset, "pkl_files")
    shard_dir = os.path.join("../Data/Processed_Data", dataset, "pkl_files/shards")
    shard_subdirs = os.listdir(shard_dir)
    shard_file_map = pickle.load(open(os.path.join(save_dir, "train_shard_file_map.pkl"), "rb"))

    i = 0
    while i < len(shard_subdirs):
        pool = mp.Pool()
        jobs = []

        for j in range(i, i + parallelism):
            if j >= len(shard_subdirs):
                break
            subdir = shard_subdirs[j]
            job = pool.apply_async(score_sentences_worker, (shard_dir, subdir, shard_file_map))
            jobs.append(job)

        for job in jobs:
            job.get()

        pool.close()
        pool.join()
        i += parallelism


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default=None)
    args.add_argument("--test_split", type=float, default=0.35)
    args.add_argument("--val_split", type=float, default=0.05)
    args.add_argument("--cpu", action = "store_true")
    args.add_argument("--parallelism", type=int, default=10)
    args.add_argument("--shard_size", type=int, default=25000, help="number of articles per shard")
    opts = args.parse_args()

    PER_TEST_FILES = opts.test_split * 100.0
    PER_VAL_FILES = opts.val_split * 100.0

    device = 'cpu' if opts.cpu else 'gpu'

    if opts.dataset == "nyt":
        get_file_list("../Data/Datasets/nyt/pair_sent_matched/", "nyt")
        get_sentences("nyt", opts.shard_size)
        get_sentence_embeddings("nyt", opts.parallelism , device)
        score_sentences("nyt", opts.parallelism)

    if opts.dataset == "cnn":
        DOMAINS = ["All"]
        get_file_list("../Data/Datasets/cnn/pair_sent_matched/", "cnn")
        get_sentences("cnn", opts.shard_size)
        get_sentence_embeddings("cnn", opts.parallelism , device)
        score_sentences("cnn", opts.parallelism)

    if opts.dataset == "cnndm":
        DOMAINS = ["All"]
        get_file_list("../Data/Datasets/cnndm/pair_sent_matched/", "cnndm")
        get_sentences("cnndm", opts.shard_size)
        get_sentence_embeddings("cnndm", opts.parallelism , device)
        score_sentences("cnndm", opts.parallelism)

    if opts.dataset == "gigaword":
        DOMAINS = ["All"]
        get_file_list("../Data/Datasets/gigaword/pair_sent_matched/", "gigaword")
        get_sentences("gigaword", opts.shard_size)
        get_sentence_embeddings("gigaword", opts.parallelism , device)
        score_sentences("gigaword", opts.parallelism)

    if opts.dataset == "ontonotes_mz":
        DOMAINS = ["All"]
        get_file_list("../Data/Datasets/ontonotes_mz/pair_sent_matched/", "ontonotes_mz")
        get_sentences("ontonotes_mz", opts.shard_size)
        get_sentence_embeddings("ontonotes_mz", opts.parallelism , device)
        score_sentences("ontonotes_mz", opts.parallelism)
        f  = re.sub(pattern, "", f)
