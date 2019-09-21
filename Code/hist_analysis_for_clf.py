import pickle
import os
import numpy as np
import numpy as np
import argparse
import pickle
import os, sys
import multiprocessing as mp
from copy import deepcopy
from multiprocessing_utils import *
from tqdm import tqdm
from extract_documents import *

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations", "All"]

def processor(file_list, predictions, JobQueue):
    hist_dct = {}
    for x in tqdm(file_list):
        dataset = x[0]
        fname = x[1]

        sentences = clean_and_process_file(dataset, fname)
        sentences = sentences["full_text"]
        clf_prob = []

        sent_not_present_cnt = 0
        for sent in sentences:
            try:
                clf_prob.append(predictions[sent][1])
            except:
                sent_not_present_cnt += 1
                continue

        hist = np.histogram(clf_prob, bins = np.arange(0, 1, 0.1))
        hist_dct[x] = hist

    JobQueue.put(hist_dct)

def writer(saveFile, JobQueue):
    hist_dct = {}
    res = JobQueue.get()
    while res != "kill":
        hist_dct.update(res)
        res = JobQueue.get()
    pickle.dump( hist_dct, open(saveFile, "wb" ))

def main(dataset, type_s):
    for domain in DOMAINS:
        predictions_file = os.path.join("../Data/Processed_Data/",
                                        dataset,
                                        domain,
                                        type_s,
                                        "predictions.pkl"
                                        )

        test_file = os.path.join("../Data/Processed_Data/",
                                        dataset,
                                        domain,
                                        "test_file_list.txt"
                                        )

        file_list = open(test_file).read().strip().split("\n")

        val_file = os.path.join("../Data/Processed_Data/",
                                        dataset,
                                        domain,
                                        "val_file_list.txt"
                                        )

        file_list.extend(open(val_file).read().strip().split("\n"))

        for i in range(0, len(file_list)):
            file_list[i] = (dataset, file_list[i])

        save_file = "../TMP/" + dataset + domain + "file_histograms_clf.pkl"

        predictions = pickle.load(open(predictions_file, "rb"))

        print("Getting classifier histograms")
        print("Total files = " + str(len(file_list)))

        multiprocessing_func(   processor, file_list, 15, [predictions], \
                                writer, [save_file]
                                )
        print("Done")

    print("Loading test data")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, required = True)
    args.add_argument("-type_s", type=str, default="importance")
    opts = args.parse_args()

    if opts.dataset in ["cnn", "gigaword", "cnndm"]:
        DOMAINS = ["All"]

    main(opts.dataset, opts.type_s)
