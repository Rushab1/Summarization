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
        if sentences == "<ERROR>":
            continue

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

def analyze_histograms(pkl_file, thresholds = [0, 0.3, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0]):
    result = {}

    hist_list = pickle.load(open(pkl_file, "rb"))
    hist = [ hist_list[h][0] for h in hist_list
                    if np.NaN not in hist_list[h][0] and
                    sum(hist_list[h][0]) != 0]

    for threshold in thresholds:
        th = int(np.ceil(threshold*10))
        imp_hist = [sum(h[th:]) for h in hist]
        sum_hist = [sum(h) for h in hist]
        per_imp = np.array(imp_hist) / np.array(sum_hist)
        per_imp_avg = np.mean(per_imp)
        # print("% of important sentences = " + str(per_imp_avg))
        result[threshold] = per_imp_avg
    return result

def main(dataset, type_s, trained_on_dataset):
    for domain in DOMAINS:
        if trained_on_dataset != None:
            predictions_file = os.path.join("../Data/Cross_Classifier_Predictions/",
                                        trained_on_dataset,
                                        dataset + ".pkl")
            print(predictions_file)
        else:
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

        if trained_on_dataset == None:
            save_file = "../TMP/" + dataset + "_" + dataset + domain + "file_histograms_clf.pkl"
        else:
            save_file = "../TMP/" + trained_on_dataset + "_" + dataset + domain + "file_histograms_clf.pkl"

        predictions = pickle.load(open(predictions_file, "rb"))

        print("Getting classifier histograms")
        print("Total files = " + str(len(file_list)))

        multiprocessing_func(   processor, file_list, 15, [predictions], \
                                writer, [save_file]
                                )
        print("Done")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, required = True)
    args.add_argument("-trained_on_dataset", default=None)
    args.add_argument("-type_s", type=str, default="importance")
    opts = args.parse_args()

    if opts.dataset in ["cnn", "gigaword", "cnndm", "ontonotes_mz"]:
        DOMAINS = ["All"]

    # main(opts.dataset, opts.type_s, opts.trained_on_dataset)

    result = {}
    for f in os.listdir("../TMP/"):
        result_f = analyze_histograms("../TMP/" + f)
        f = f.replace(".pkl", "")
        result[f] = result_f
        print(f)
        print(result_f)
        print("\n")
    pickle.dump(result, open("./hist_analysis_for_clf.pkl", "wb"))
