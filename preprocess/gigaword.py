import os
import tqdm
import sys
import re
from shutil import copyfile
import random
import argparse
import json
from IPython import embed
STORIES_DIR = "../Data/Datasets/gigaword/stories/"
PAIR_SENT_DIR = "../Data/Datasets/gigaword/pair_sent_matched/All"
JSON_DIR = "../Data/Datasets/gigaword/json_dir/All"

if not os.path.exists("../Data/Datasets/gigaword/pair_sent_matched"):
    os.mkdir("../Data/Datasets/gigaword/pair_sent_matched")

if not os.path.exists(JSON_DIR):
    os.mkdir(JSON_DIR)

if not os.path.exists(PAIR_SENT_DIR):
    os.mkdir(PAIR_SENT_DIR)

if not os.path.exists(STORIES_DIR):
    os.mkdir(STORIES_DIR)

random.seed(1234)
join = os.path.join

def format_file(f):
    article_sentences = []
    abstract_sentences = []
    
    f = f.strip()
    pattern = re.compile("<DOC id=\"(.*?)\".*?>", re.DOTALL)
    match = re.match(pattern, f)
    assert(match != None)
    fname = match.group(1)

    pattern = re.compile("<HEADLINE>\n(.*)\n</HEADLINE>", re.DOTALL)
    match = re.findall(pattern, f)
    try:
        assert(match != [] )
    except:
        return fname, [], []

    print(match)
    assert(len(match) == 1)
    headline = match[0]

    pattern = re.compile(".*<TEXT>\n(.*)\n</TEXT>.*", re.DOTALL)
    match = re.match(pattern, f)
    assert(len(match.groups()) == 1)
    text = match.group(1)

    pattern = re.compile("<P>(.*?)</P>", re.DOTALL)
    sentences = re.findall(pattern, text)

    for s in sentences:
        s = s.strip()
        s = re.sub("\n", " ", s)
        article_sentences.append(s)

    abstract_sentences.extend(headline.strip().split("\n"))
    return fname, article_sentences, abstract_sentences

def preprocess(dataDir, max_files = None):
    subdirs = os.listdir(dataDir)

    for dir in subdirs:
        file_list = os.listdir(join(dataDir, dir))
        random.shuffle(file_list)
        files = []
        
        print(file_list, dataDir, dir)
        for fname in file_list:
            print(dir, fname)
            f = open(join(dataDir, dir, fname)).read().strip()

            pattern = re.compile("<DOC id.*?>.*?</DOC>", re.DOTALL)
            # f = re.sub("<DOC id=\".*", "&__new_document__&", f)
            # f = f.split("&__new_document__&")
            f = re.findall(pattern, f)
            if f[0] == "":
                del f[0]
            files.extend(f)

            print(len(f))
            if max_files != None and len(files) > 10*max_files:
                break
    
        if max_files == None:
            random.shuffle(files)
        else:
            files = random.sample( files, max_files)

        print(len(files))

        for i in range(0, len(files)):
            fname, article_sentences, abstract_sentences = format_file(files[i])
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
    args.add_argument("-max_files", type=int, default = 10000)
    opts = args.parse_args()

    preprocess(STORIES_DIR, opts.max_files)


