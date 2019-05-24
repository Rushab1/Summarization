import os
import pickle
import argparse
import nltk
from val_test_predictions import normalize

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
