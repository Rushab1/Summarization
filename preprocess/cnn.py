#preprocess cnn dataset
import os
import json
from tqdm import tqdm
import nltk

STORIES_DIR = "../Data/Datasets/cnn/stories"
PAIR_SENT_DIR = "../Data/Datasets/cnn/pair_sent_matched"
JSON_DIR = "../Data/Datasets/cnn/json_dir"

if not os.path.exists("../Data/Datasets/cnn/pair_sent_matched"):
    os.mkdir("../Data/Datasets/cnn/pair_sent_matched")

if not os.path.exists(JSON_DIR):
    os.mkdir(JSON_DIR)

stories = os.listdir(STORIES_DIR)

for story in tqdm(stories):
    article_sentences = []
    abstract_sentences = []

    fname = os.path.join(STORIES_DIR, story)
    f = open(fname).read().split("\n")

    i = 0
    while i < len(f):
        if f[i].strip() == "":
            i += 1
            continue

        if f[i].strip().lower() == "@highlight":
            i += 1
            while f[i].strip() == "":
                i += 1

            local_sentences = nltk.sent_tokenize(f[i])
            abstract_sentences.extend(local_sentences)
            i += 1

        else:
            local_sentences = nltk.sent_tokenize(f[i])
            article_sentences.extend(local_sentences)
            i += 1

    dct = { "article":"\n".join(article_sentences),
            "abstract": "\n".join(abstract_sentences)
            }

    json_file = open(os.path.join(JSON_DIR, story.replace(".story", ".json")), "w")
    json.dump(dct, json_file)

    l = []
    for art_sent in article_sentences:
        for abs_sent in abstract_sentences:
            dct = { "source": abs_sent,
                    "target": art_sent
                    }
            l.append(dct)

    json_file = open(os.path.join(PAIR_SENT_DIR, story.replace(".story", ".json")), "w")
    json.dump(l, json_file)
