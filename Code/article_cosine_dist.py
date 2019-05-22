import pickle
import numpy as np
import random
import os
import multiprocessing as mp
from extract_documents import *

def a(year, q):
    try:
        cos = []

        max_score_for_article = {}
        for i in np.arange(0, 1.01, 0.1):
            i = round(i, 1)
            max_score_for_article[i] = {"count":0, "examples":[]}

        print("Loading " + year)
        p = pickle.load(open("./USE_Data_files/" + year + "/Summaries_pairs_dict.pkl"))

        p = p[year]
        cnt = 0
        cnt1 = 0
        for k in p:
            pairs = p[k]['pairs']

            if len(pairs) == 0:
                continue
            done = {}

            for i in range(len(pairs)-1, -1, -1):
                pair = pairs[i]
                
                if pair['summary_sent'] in done:
                    continue
                cos.append(pair['cosine_sim'])
                done[pair['summary_sent']] = 1


            max_cos = pairs[len(pairs)-1]['cosine_sim']
            key = np.floor(max_cos * 10)/10.0


            key = round(key, 1)

            max_score_for_article[key]['count'] += 1

            if len(max_score_for_article[key]['examples']) < 2:
                orig_content = clean_and_process_file(k, year)

                if type(orig_content) != dict:
                    continue
                
                full_text = orig_content['full_text']
                article_sentences = []

                for i in range(len(pairs)-1, -1, -1):
                    article_sentences.append(\
                            str(round(pairs[i]['cosine_sim'],2)) + \
                            " :: " + pairs[i]['sent'])


                max_score_for_article[key]["examples"].append({
                        "summary" : p[k]['summary'],
                        "article" : article_sentences,
                        "full_text": full_text
                        })
        print(year + " Done")
        q.put([cos, max_score_for_article])
    except Exception as e:
        print("_______________________")
        print(e)
        print("_______________________")


def b(q):
    cos = []
    max_score_for_article = {}
    for i in np.arange(0, 1.01, 0.1):
        i = round(i, 1)
        max_score_for_article[i] = {"count":0, "examples":[]}

    while 1:
        res = q.get()
        if res == 'kill':
            print("dumping")
            pickle.dump(cos, open("./train_test_data/just_cosine_scores_summaries.pkl", "wb"))
            pickle.dump(max_score_for_article, open("./train_test_data/article_cosine_dist.pkl", "wb"))
            print("done")
            break
        
        cos.extend(res[0])
        res1 = res[1]

        for i in np.arange(0, 1.01, 0.1):
            i = round(i, 1)
            max_score_for_article[i]['count'] += res1[i]['count']
            max_score_for_article[i]['examples'].extend(res1[i]["examples"])

pool = mp.Pool()
manager = mp.Manager()
q = manager.Queue()
jobs = []

years = os.listdir("./USE_Data_files/")
# years = ['1996']

writer = pool.apply_async(b, (q,))
for year in years:
    job = pool.apply_async(a, (year, q,))
    jobs.append(job)


for job in jobs:
    job.get()

q.put("kill")
writer.get()
pool.close()
pool.join()
