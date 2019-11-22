import os
import pickle
import argparse
import nltk
# from val_test_predictions import normalize
from extract_documents import *
from tqdm import tqdm
import numpy as np
import random

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations", "All"]
MAX_ARTICLES = 1000

#each of these three files have len(file_list) lines, each file representing an article
#Every sentence starts with <t> and ends with </t>
def analyze_lead(dataset, file_list, predictions_dct, save_file, min_len = 100):
    analyze_dict = {}
    for threshold in tqdm([0, 0.6, 0.65, 0.7]):
        cnt = 0
        cnt2 = 0
        removed = []

        new_article = []
        new_pred = []
        new_abs = []
        for i in tqdm(range(0, len(file_list))):
            fname = file_list[i]

            dct = clean_and_process_file(dataset, fname)
            article = dct["full_text"]
            abstract = dct["abstract"]

            len_lead = 0
            num_removed = 0
            num_sentences = 0
            local_pred = []
            for sent in article:
                cnt2 += 1
                flag = False
                if sent not in predictions_dct:
                    cnt += 1
                    flag = True
                else:
                    # print(predictions_dct[sent][1])
                    if predictions_dct[sent][1] > threshold:
                        len_lead += len(sent.split())
                        local_pred.append(sent)
                    else:
                        num_removed += 1
                num_sentences += 1

                #For Lead - 3 Analysis
                # if num_sentences == 3:
                    # break
                if len_lead > min_len:
                    break
            removed.append(num_removed)

            if num_removed > 0:
                new_pred.append(" ".join(local_pred))
                new_abs.append(" ".join(abstract))

                tmp = " ".join(local_pred).split()
                tmp1 = " ".join(article).split()
                new_article.append( " ".join( tmp1 [: len(tmp)]))

        f = open("./Lead/" + dataset + ".pred." + str(threshold) + ".txt", "w")
        f.write("\n".join(new_pred))
        f.close()

        g = open("./Lead/" + dataset + ".abstracts." + str(threshold) + ".txt" , "w")
        g.write("\n".join(new_abs))
        g.close()

        g = open("./Lead/" + dataset + ".article." + str(threshold) + ".txt" , "w")
        g.write("\n".join(new_article))
        g.close()

        analyze_dict[threshold] = removed
        print("\n\n")
        print(threshold, np.mean(removed))
        print("\n\n")

    pickle.dump(analyze_dict, open(save_file, "wb"))
    return analyze_dict

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default="nyt")
    args.add_argument("-type_s", type=str, default="importance") #Do not specify if you want both classifiers
    args.add_argument("-split", type=str, default="test") #valid/test
    args.add_argument("-domain", type=str, default="All") #Business/Sports/Science/USIntlRelations/All=None
    args.add_argument("-len", type=int, default = 100) #Business/Sports/Science/USIntlRelations/All=None
    opts = args.parse_args()

    if opts.split == "valid":
        opts.split = "val"

    if opts.domain == "None":
        opts.domain = "All"

    dataset = opts.dataset
    split = opts.split
    domain = opts.domain
    type_s = opts.type_s

    file_list_file = os.path.join("../Data/Processed_Data/", dataset, domain, split + "_file_list.txt")
    file_list = open(file_list_file).read().strip().split("\n")
    random.shuffle(file_list)

    MAX_ARTICLES = min(MAX_ARTICLES, len(file_list))
    file_list = file_list[:MAX_ARTICLES]

    load_file = os.path.join("../Data/Processed_Data/", dataset, domain, type_s, "predictions.pkl")
    predictions_dct = pickle.load(open(load_file, "rb"))

    save_file = "./Results/lead_analyze_" + dataset + "_" + split + "_" + str(opts.len) + "_" + domain + ".txt" 
    analyze_lead(dataset, file_list, predictions_dct, save_file, opts.len)
