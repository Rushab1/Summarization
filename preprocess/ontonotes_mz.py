import os
from tqdm import tqdm
import sys
import re
from shutil import copyfile
import random
import argparse
import json
from IPython import embed
STORIES_DIR = "../Data/Datasets/ontonotes/stories/data/files/data/english/annotations/mz/sinorama/10/"
PAIR_SENT_DIR = "../Data/Datasets/ontonotes_mz/pair_sent_matched/All"
JSON_DIR = "../Data/Datasets/ontonotes_mz/json_dir/All"

if not os.path.exists("../Data/Datasets/ontonotes_mz/pair_sent_matched"):
    os.mkdir("../Data/Datasets/ontonotes_mz/pair_sent_matched")

if not os.path.exists("../Data/Datasets/ontonotes_mz/json_dir"):
    os.mkdir("../Data/Datasets/ontonotes_mz/json_dir")

if not os.path.exists(JSON_DIR):
    os.mkdir(JSON_DIR)

if not os.path.exists(PAIR_SENT_DIR):
    os.mkdir(PAIR_SENT_DIR)

if not os.path.exists(STORIES_DIR):
    os.mkdir(STORIES_DIR)

random.seed(1234)
join = os.path.join

#format the .name versions of the file
def format_file(f):
    article_sentences = []
    abstract_sentences = []
    
    f = f.strip()
    f = re.sub("<ENAMEX.*?>", "", f)
    f = re.sub("</ENAMEX>", "", f)

    f = f.strip().split("\n")
    del f[0] #<DOC> tag
    del f[-1] #</DOC> tag

    #Headline now on line 1
    headline = f[0]
    article_sentences = f
    abstract_sentences.extend(headline.strip().split("\n"))
    return article_sentences, abstract_sentences

def preprocess(dataDir, max_files = None):
    file_list = os.listdir(dataDir)
    random.shuffle(file_list)
    files = []
    
    print(len(file_list))
    i = 0
    while i < len(file_list):
        fname = file_list[i]
        if not fname.endswith(".name"):
            del file_list[i]
            continue
        i += 1

    if max_files != None:
        file_list = file_list[: max_files]

    for fname in tqdm(file_list):
        f = open(join(dataDir, fname)).read().strip()
        article_sentences, abstract_sentences = format_file(f)

        if len(article_sentences) == 0:
            continue

        dct = { "article":"\n".join(article_sentences),
                "abstract": "\n".join(abstract_sentences)
                }

        json_file = open(os.path.join(JSON_DIR, fname + ".json"), "w")
        json.dump(dct, json_file)

        l = []
        for art_sent in article_sentences:
            for abs_sent in abstract_sentences:
                dct = { "source": abs_sent,
                        "target": art_sent
                        }
                l.append(dct)

        json_file = open(os.path.join(PAIR_SENT_DIR, fname + ".json"), "w")
        json.dump(l, json_file)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-max_files", type=int, default = 1000000)
    opts = args.parse_args()

    preprocess(STORIES_DIR, opts.max_files)


