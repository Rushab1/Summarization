import os
import sys
import pickle
import re
import nltk
import json

def clean_and_process_file(dataset, fname):
    if dataset == "nyt":
        return clean_and_process_nyt_file(fname)
    elif dataset == "cnn" or dataset == "cnndm":
        return clean_and_process_cnn_file(fname, dataset)
    else:
        return clean_and_process_cnn_file(fname, dataset)

def clean_and_process_cnn_file(fname, dataset):
    try:
        dir_orig = os.path.join("../Data/Datasets", dataset, "json_dir/All")
        fpath = os.path.join(dir_orig, fname.split("/")[-1])
        story = json.load(open(fpath))
    except Exception as e:
        # print(e)
        return "<ERROR>"
        pass
        # fpath = fname
        # story = json.load(open(fpath))

    #already tokenized by nltk sent_tokenize during preprocessings
    try:
        full_text = story['article'].split("\n")
    except:
        print(fpath);

    abstract = story['abstract'].split("\n")

    return_dct = {
            "abstract": abstract,
            "full_text": full_text
            }
    return return_dct

def clean_and_process_nyt_file( fname):
    dir_orig = "../Data/Datasets/nyt/orig_dataset/"
    f = fname.split("/")[-1]
    f = open(dir_orig + f.replace(".json", ".xml"))
    f = f.read()

    abstract = f.split('<abstract>')
    if len(abstract) == 1:
        # continue
        return ""

    abstract = abstract[1].split('</abstract>')
    f = abstract[1]
    abstract = abstract[0]
    # abstract = re.sub(";.*?$", "", abstract)
    abstract = abstract.replace("<p>", "")
    abstract = abstract.replace("</p>", "")

    #Clean abstract
    abstract = abstract.strip()
    abstract = abstract.split(';')

    tmp = []
    for para in abstract:
        tmp.extend(nltk.sent_tokenize(para))
    abstract = tmp



    lead_paragraph = f.split("<block class=\"lead_paragraph\">")

    lead_paragraph = lead_paragraph[0].split("</block>")[0]
    lead_paragraph = lead_paragraph.replace("<p>", "")
    lead_paragraph = lead_paragraph.replace("</p>", "")

    full_text = f.split("<block class=\"full_text\">")
    full_text = full_text[1]
    full_text = full_text.replace("<p>", "\n")
    full_text = full_text.replace("</p>", "\n")
    pattern = re.compile("\<\/block\>.*", re.DOTALL)
    full_text = re.sub( pattern, "", full_text)
    full_text = re.sub('\'\'', r'"', full_text)
    full_text = re.sub('([.,!?()\'"])', r' \1 ', full_text)
    full_text = full_text.strip()
    full_text = re.sub("\t", " ", full_text)
    full_text = re.sub(" [ ]*", " ", full_text)

    # full_text = full_text.decode("utf-8")
    # full_text = full_text.encode("ascii", "ignore")
    # full_text = full_text.split("\n")
    # full_text = full_text.replace("\n", " ")

    full_text = full_text.replace(" .", ".")
    full_text = full_text.replace("\n[\n]*", "\n")
    full_text = full_text.split("\n")

    text = []
    for para in full_text:
        text.extend(nltk.sent_tokenize(para))

    full_text = text
    # full_text = nltk.sent_tokenize(full_text)


    # for i in range(0, len(full_text)):
        # full_text[i] = full_text[i].strip()
    # full_text = f.split("</block>")

    return_dct = {
            "abstract": abstract,
            "lead_paragraph": lead_paragraph,
            "full_text": full_text
            }
    return return_dct
