import os 
import pickle
import argparse
import numpy as np
from copy import deepcopy
import tensorflow as tf
import tensorflow_hub as tf_hub
import multiprocessing as mp
from extract_documents import * 
from gensim.summarization.summarizer import summarize
import nltk

THRESHOLD = 0.6

def remove_unwanted_from_from_all_articles(articles, model):
    article_sentences = []
    cleaned_articles = []
    cnt = 0

    for article in articles:
        article = article.decode('utf-8')
        article = article.encode('ascii', 'ignore')
        text = nltk.sent_tokenize(article)
        article_sentences.extend(text)
    
    article_sentences_dct = {}
    article_sentences = list(set(article_sentences))

    print("Removing unwanted")
    _, imp_labels = remove_unwanted(article_sentences, model)
    print("Done")

    for i in range(0, len(article_sentences)):
        article_sentences_dct[article_sentences[i]] = imp_labels[i]

    total = 0
    thrown_away = 0

    for article in articles:
        article = article.decode('utf-8')
        article = article.encode('ascii', 'ignore')
        text = nltk.sent_tokenize(article)

        cleaned_text = []
        for sent in text:
            total += 1
            if article_sentences_dct[sent] == 1:
                cleaned_text.append(sent)
            else:
                thrown_away += 1

        cleaned_articles.append(" ".join(cleaned_text))
    return "\n".join(cleaned_articles)

def create_embeddings(text):
    print("Loading USE tf_hub Module")
    embed_module = tf_hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    print("Done")

    print("Creating session")
    session = tf.Session()
    session.run([ tf.global_variables_initializer(), tf.tables_initializer()  ])
    print("Done")

    #Check if embeddings present else create new embeddings
    embeddings = session.run(embed_module(text))
    embeddings_dct = {}

    for i in range(0, len(text)):
        embeddings_dct[text[i]] = embeddings[i]
    
    pickle.dump({"dct":embeddings_dct, "list":embeddings, "text":text}, open("train_test_data/embeddings_dct.pkl", "w"))
    return embeddings

#Create embeddings for test articles and remove unimportant Sentences from a SINGLE article
def remove_unwanted(text, model):
    global THRESHOLD
    if type(text) != list:
        text = text.split("\n")
    
    print(":Started removing unwanted")

    print("Getting embeddings")
    if(os.path.exists("./train_test_data/embeddings_dct.pkl")):
        try:
            embeddings_dct = pickle.load(open("train_test_data/embeddings_dct.pkl"))["dct"]
            embeddings = []
            for sent in text:
                embeddings.append(embeddings_dct[sent])
            embeddings = np.array(embeddings)
        except Exception as e:
            print("Error while loading embeddings: Creating new embeddings")
            print("___________\n" + str(e) + "___________")
            embeddings = create_embeddings(text)
    else:
        embeddings = create_embeddings(text)
        
    mu = model[1]
    sig = model[2]
    mu.resize(512, 1)
    sig.resize(512, 1)
    embeddings = embeddings.transpose()
    embeddings = (embeddings-mu)/sig
    embeddings = embeddings.transpose()

    print("Predicting Importance")
    try:
        text_predict_dct = pickle.load(open("./train_test_data/predictions.pkl"))
        text_predict = []
        for sent in text:
            text_predict.append(text_predict_dct[sent])
    
    except:
        print("Predicting Importance: REDOING IT")
        print(embeddings.shape)
        text_predict = model[0].predict_proba(embeddings)
        text_predict_dct = {}

        for i in range(0,len(text)):
            text_predict_dct[text[i]] = text_predict[i]

        pickle.dump(text_predict_dct, open("./train_test_data/predictions.pkl", "w"))

    print("Done")

    assert(len(text_predict)) == len(embeddings)
    
    text_cleaned = []
    text_predict_bool = []
    for i in range(0, len(text)):
        # if text_predict[i] == 1:
        if text_predict[i][1] > THRESHOLD:
            text_cleaned.append(text[i])
            text_predict_bool.append(1)
        else:
            text_predict_bool.append(0)

    text_cleaned = "\n".join(text_cleaned)
    print( "Done removing unwanted")
    return text_cleaned, text_predict_bool#, text_predict_proba

def main1(modelfile, type_s, test=True, valid = False):
    print("Loading Model")
    model = pickle.load(open(modelfile))
    print("Done")
    
    if test:
        articles = open("./train_test_data/untouched_test_data/" + type_s + "_articles.txt").read().strip().split("\n")
        cleaned_articles = remove_unwanted_from_from_all_articles(articles, model)

        f = open("./train_test_data/untouched_test_data/cleaned_" + type_s + "_articles.txt", "w")
        f.write(cleaned_articles)
        f.close()

    if valid:
        articles = open("./train_test_data/untouched_valid_data/" + type_s + "_articles.txt").read().strip().split("\n")
        cleaned_articles = remove_unwanted_from_from_all_articles(articles, model)

        f = open("./train_test_data/untouched_valid_data/cleaned_" + type_s + "_articles.txt", "w")

        # print("______________________________________________")
        # print(cleaned_articles[:1000])
        # print("______________________________________________")
        f.write(cleaned_articles)
        f.close()

        # fname = "./train_test_data/untouched_valid_data/cleaned_" + type_s + "_articles.txt"
        # print("\n" + fname + "\n")
        # f = open(fname, "r").read().split("\n")
        # print(f[:3])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-type_s", type=str, default="by_sentence")
    args.add_argument("-threshold", type=float, default=0.6)
    args.add_argument("-test", type=int , default=1)
    args.add_argument("-valid", type=int, default=0)

    opts = args.parse_args()

    type_s = opts.type_s

    global THRESHOLD
    THRESHOLD = opts.threshold

    main1("./modelfiles/" + type_s + "/model.pkl", type_s, opts.test, opts.valid)
