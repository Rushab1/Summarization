import os
import json
from tqdm import tqdm

PAIR_SENT_DIR = "../Data/Datasets/cnn/pair_sent_matched"

if not os.path.exists("../Data/Datasets/cnndm/json_dir"):
    os.mkdir("../Data/Datasets/cnndm/json_dir")
    os.mkdir("../Data/Datasets/cnndm/json_dir/All")

if not os.path.exists("../Data/Datasets/cnndm/pair_sent_matched"):
    os.mkdir("../Data/Datasets/cnndm/pair_sent_matched")
    os.mkdir("../Data/Datasets/cnndm/pair_sent_matched/All")

def preprocess(split = "test"):
    articles = open("../Data/Datasets/cnndm/" + split + ".txt.src").read().split("\n")
    abstracts = open("../Data/Datasets/cnndm/" + split + ".txt.tgt.tagged").read().split("\n")

    for i in tqdm(range(0, len(articles))):
        articles[i] = articles[i].replace( "\n", " " )
        articles[i] = articles[i].replace( " . ", ".\n" )

        abstracts[i] = abstracts[i].replace("<t>", " ")
        abstracts[i] = abstracts[i].replace("</t>", "\n")
        abstracts[i] = abstracts[i].strip()

        dct = {"article": articles[i],
                "abstract": abstracts[i]
                }

        article_sentences = articles[i].split("\n")
        abstract_sentences = abstracts[i].split("\n")

        pair_sent = []
        for tar in article_sentences:
            for src in abstract_sentences:
                pair_sent.append({"source": src, "target":tar})

        json.dump(dct, open("../Data/Datasets/cnndm/json_dir/All/" + split + "_" + str(i) + ".json", "w") )
        json.dump(pair_sent, open("../Data/Datasets/cnndm/pair_sent_matched/All/" + split + "_" + str(i) + ".json", "w") )

preprocess("train")
preprocess("test")
preprocess("val")
