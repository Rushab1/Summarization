import os
import pickle
import argparse
import nltk
from val_test_predictions import normalize
from extract_documents import *
from tqdm import tqdm

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations", "All"]

#article = list of article sentences, example - tokenized by nltk.sent_tokenizer (order of sentences must be preserved)
#embeddings_dct = dictionary of the form <sentence, its embedding>. Can use get_sentence_embeddings.py for this
#model = <sklearn_model, mu, sig> #Can use train.py for this (mu, sig) are needed to normalize the embeddings, else pass (0,1)
def clean(article, embedddings_dct, model, threshold = 0.6):
    embeddings = []
    for sent in article:
        embeddings.append(embedddings_dct[sent])

    sklearn_model = model[0]
    mu = model[1]
    sig = model[2]

    embeddings = normalize(embeddings, mu, sig)
    prob = sklearn_model.predict_proba(embeddings)

    cleaned_article = []
    for i in range(0, len(article)):
        if prob[i] >= threshold:
            cleaned_article.append(article[i])

    return cleaned_article

#save_dir = directory containing three files - orig.txt, cleaned.txt, abstracts.txt
#each of these three files have len(file_list) lines, each file representing an article
#Every sentence starts with <t> and ends with </t>
def create_cleaned_files_list(dataset, file_list, predictions_dct, save_dir, threshold = 0.6):
    orig = open(os.path.join(save_dir, "orig.txt"), "w")
    cleaned = open(os.path.join(save_dir, "cleaned.txt"), "w")
    abstracts = open(os.path.join(save_dir, "abstracts.txt"), "w")

    cnt = 0
    cnt2 = 0
    for i in tqdm(range(0, len(file_list))):
        fname = file_list[i]

        dct = clean_and_process_file(dataset, fname)
        article = dct["full_text"]
        abstract = dct["abstract"]

        for sent in article:
            cnt2 += 1
            flag = False
            if sent not in predictions_dct:
                cnt += 1
                flag = True
            else:
                # print(predictions_dct[sent][1])
                if predictions_dct[sent][1] > threshold:
                    flag = True

            sent = sent.replace("\n", " ")
            sent = " <t> " + sent + " </t> "
            orig.write(sent)
            if flag:
                cleaned.write(sent)

        for sent in abstract:
            sent = sent.replace("\n", " ")
            sent = " <t> " + sent + " </t> "
            abstracts.write(sent)

        if i != len(file_list) - 1:
            orig.write("\n")
            cleaned.write("\n")
            abstracts.write("\n")

    print(cnt, cnt2)
    orig.close()
    cleaned.close()
    abstracts.close()
    orig = open(os.path.join(save_dir, "orig.txt")).read().split()
    cleaned = open(os.path.join(save_dir, "cleaned.txt")).read().split()
    print("Reduction = " + str( 100.0 * len(cleaned) / len(orig)))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default="nyt")
    args.add_argument("-type_s", type=str, default=None) #Do not specify if you want both classifiers
    args.add_argument("-split", type=str, default="valid") #valid/test
    args.add_argument("-domain", type=str, default=None) #Business/Sports/Science/USIntlRelations/All=None
    args.add_argument("-save_dir", type=str, default="../TMP/") #Business/Sports/Science/USIntlRelations/All=None
    args.add_argument("-threshold", type=float, default = 0.7) #Business/Sports/Science/USIntlRelations/All=None
    opts = args.parse_args()

    if opts.split == "valid":
        opts.split = "val"

    if opts.domain == "None":
        opts.domain = "All"

    dataset = opts.dataset
    split = opts.split
    domain = opts.domain
    type_s = opts.type_s
    save_dir = opts.save_dir
    threshold = opts.threshold

    file_list_file = os.path.join("../Data/Processed_Data/", dataset, domain, split + "_file_list.txt")
    file_list = open(file_list_file).read().strip().split("\n")

    load_file = os.path.join("../Data/Processed_Data/", dataset, domain, type_s, "predictions.pkl")
    predictions_dct = pickle.load(open(load_file, "rb"))
    create_cleaned_files_list(dataset, file_list, predictions_dct, save_dir, threshold)
