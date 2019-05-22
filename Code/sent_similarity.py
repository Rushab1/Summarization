import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as tf_hub
import os, re, sys
import numpy as np
import json
import pickle
import random
import multiprocessing as mp
from scipy.stats import spearmanr, pearsonr
from tensorflow.python.client import device_lib
from tqdm import tqdm

data_dir = "/nlp/users/yinfeiy/informative_summary/data/full/pair_sent_matched/"
data_dir = "../Data/Datasets/nyt/pair_sent_matched/"
PER_EXCLUDED_FILES = 40.0 #Total files = 58419 in Business
embed_module = None
session = None

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def gpu_is_available():
    gpu_list = get_available_gpus()
    return True if len(gpu_list)>0 else False

#Makes a file of all training and test file locations for every year given the news type
#Saved at USE_Data_files/<Year>/train_file_list.txti and USE_Data_files/<year>/test_file_list.txt
def get_file_list(data_dir, year):
    file_list = []
    test_file_list = []

    for root, subdir, files in os.walk(data_dir + year):
        for filename in files:
            file_list.append(root + "/" + filename)

    file_list = random.sample(file_list, len(file_list))

    num_files = len(file_list)
    num_excluded_files = PER_EXCLUDED_FILES/100.0 * num_files
    num_excluded_files = int( num_excluded_files )

    test_file_list = file_list[-num_excluded_files:]
    file_list = file_list[ : (num_files - num_excluded_files) ]

#    print len(test_file_list), len(file_list)

    f = open("USE_Data_files/" + year + "/test_file_list.txt", "w")
    f.write("\n".join(test_file_list))
    f.close()

    f = open("USE_Data_files/" + year +"/train_file_list.txt", "w")
    f.write("\n".join(file_list))
    f.close()

def cos_similarity(s1, s2):
    sim = np.dot(s1, s2)
    sim /= np.linalg.norm(s1)
    sim /= np.linalg.norm(s2)
    return sim

def angular_similarity(s1, s2):
    sim = cos_similarity(s1, s2)
    sim = np.arccos(sim)
    sim = (1. - sim) / np.pi
    return sim

def get_sentences(data_dir):
    print("Creating Manager, JobQueue and Pool")
    manager = mp.Manager()
    JobQueue = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    print("Number of Cores = " + str(mp.cpu_count()))
    print("Calling writer")
    writer = pool.apply_async(sentences_writer, (JobQueue,))

    print("Creating jobs")
    jobs = []
    for i in os.listdir(data_dir):
        try:
            args =  (data_dir, i, JobQueue, )
            job = pool.apply_async(sentences_worker, args)
        except Exception as e:
            print(e)
            sys.exit(0)
    print("Jobs created")

    for job in jobs:
        job.get()

    print("Closing pool")
    pool.close()
    pool.join()

    print("DONE")

#Get a list of all sentences, get embeddings and save them in USE_Data_files/<year>/Sentences_dict.pkl
#The list of sentences in saved in Sentences.pkl
def sentences_worker(data_dir, year, JobQueue):
    print("Extracting sentences")

    year = str(year)
    Sentences = []
    local_sent = []

    if data_dir[-1] == "/":
        data_dir = data_dir[:-1]

    for root, subdir, files in os.walk(data_dir +"/" + year):
        for filename in files:
            f = open(root + "/" + filename)
            try:
                f_pairs = json.load(f)
            except:
                print("Cannot load json: " + f + ": Skipping")
                continue

            for dct in f_pairs:
                source = dct['source']
                target = dct['target']
                local_sent.append(source)
                local_sent.append(target)

            local_sent = list(set(local_sent))
            Sentences.extend(local_sent)
            tmp = str(len(local_sent))
            local_sent = []


        sys.stdout.write(year + ":" + tmp + ":" + \
                            str(len(Sentences)))
        sys.stdout.flush()

    try:
        if not os.path.exists("./USE_Data_files/"):
            os.mkdir("./USE_Data_files")
    except Exception as e:
        print("USE Error: " + e)

    try:
        if not os.path.exists("USE_Data_files/" + year):
            os.mkdir("./USE_Data_files/" + year)
    except Exception as e:
        print("USE Error - 2 " + e)

    save_file = open("./USE_Data_files/" + str(year) + \
                "/Sentences.pkl", "wb")
    pickle.dump(Sentences, save_file)

    print("Creating Embeddings")
    Sentences_dict = {}
    global embed_module
    if embed_module == None:
        print("Loading USE tf_hub Module")

        if gpu_is_available():
            embed_module = tf_hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        else:
            embed_module = tf_hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

    #Initialize
    print("TF initialize")
    global session
    if session == None:
        if not gpu_is_available():
            session = tf.Session()
        else:
            config = tf.ConfigProto()
            config.graph_options.rewrite_options.shape_optimization = 2
            session = tf.Session(config=config)
            session.run([ tf.global_variables_initializer(), tf.tables_initializer()  ])

    n = len(Sentences)
    if gpu_is_available():
        h = 100000
    else:
        h = 200000

    for i in tqdm(range(0, 1 + int(n/h))):
        s = i*h
        e = min((i+1)*h, n)

        if s>= n:
            break

        embeddings = session.run(embed_module(Sentences[s:e]))

        for j in range(s, e):
            Sentences_dict[Sentences[j]] = embeddings[j-s]

    save_file = open("./USE_Data_files/" + year + "/" + \
            "Sentences_dict.pkl" , "wb")
    # print(Sentences_dict.keys())
    try:
        pickle.dump(Sentences_dict, save_file)
    except Exception as e:
        print(e)

    print("DONE Year: " + year)
    JobQueue.put(year)

#Has no use
def sentences_writer(JobQueue):
    print("Entering writer")
    try:
        # while 1:
            m = JobQueue.get()
            print(m)
            # if m == "kill":
                # break
    except Exception as e:
        print(e)

################################################################
################################################################
def score_summaries(data_dir, year, sent_to_embed=None):
    year = str(year)
    print("Scoring Summaries for Year " + year)

    if sent_to_embed == None:
        sent_to_embed = pickle.load(open("./USE_Data_files/" + \
            year + "/Sentences_dict.pkl", "rb"))
    Summaries = {}

    cnt = 0
    if not data_dir.endswith("/"):
        data_dir += "/"

    #Loop over all files again and get all source-target pairs
    print("YEAR: " + year + " : Looping over all files")
    Summaries[year] = {}

    file_list = open("USE_Data_files/" + year + \
            "/train_file_list.txt").read().strip().split("\n")

    for filename in file_list:
            cnt += 1
            # f = open(root + "/" + filename)
            f = open(filename)

            try:
                f_pairs = json.load(f)
            except:
                print("Cannot load JSON file: " + filename)
                continue

            filename = filename.split("/")[-1]
            Summaries[year][filename] = {}
            loc_summary = Summaries[year][filename]


            summary = []
            for dct in f_pairs:
                summary.append(dct['source'])
            summary = list(sorted(set(summary), key=summary.index))

            loc_summary['summary'] = " ".join(summary)
            loc_summary['pairs'] = []

            for dct in f_pairs:
                source = dct['source']
                target = dct['target']
                jacana_sim = dct["score"]
                sureAlign = dct["sureAlign"]
                possibleAlign = dct["possibleAlign"]

                source_embed = sent_to_embed[source]
                target_embed = sent_to_embed[target]

                sim = cos_similarity(source_embed,target_embed)
                ang_sim = angular_similarity(source_embed, \
                                        target_embed)

                tmp_dct = {}
                tmp_dct['sent'] = target
                tmp_dct['summary_sent'] = source
                tmp_dct['cosine_sim'] = sim
                tmp_dct['angular_sim'] = ang_sim
                tmp_dct['jacana_sim'] = jacana_sim
                tmp_dct['sureAlign'] = sureAlign
                tmp_dct['possibleAlign'] = possibleAlign
                loc_summary['pairs'].append(tmp_dct)

            t = loc_summary['pairs']
            loc_summary['pairs'] = sorted(t, \
                    key=lambda t: t['cosine_sim'])


    print("YEAR: " + year + "saving data to " +  "USE_Data_files/" +year + "/Summaries_pairs_dict.pkl")
    try:
        os.system("mv USE_Data_files/" +year + "/Summaries_pairs_dict.pkl " + "USE_Data_files/" + year + "/backup_Summaries_pairs_dict.pkl")
    except:
        pass

    save_file = open("USE_Data_files/" +year + "/Summaries_pairs_dict.pkl", "wb")
    pickle.dump(Summaries, save_file)
    print("YEAR : " + year  +": DONE")

################################################################
################################################################
#(cosine) Score pairs of sentences one from the article and the other from the summary by year and news type
#Save to USE_Data_files/<year>/Sentences_pairs_dict.pkl
def score_sentences(data_dir, year):
    year = str(year)
    print("Scoring Sentences for Year " + year)
    sent_to_embed = pickle.load(open("./USE_Data_files/" + year + \
            "/Sentences_dict.pkl", "rb"))
    Sentences = {}
    sent_to_ind = {}

    sent_keys = sent_to_embed.keys()
    for i in range(0, len(sent_keys)):
        sent = sent_keys[i]
        sent_to_ind[sent] = i
        Sentences[i] = {}
        Sentences[i]['sentence'] = sent
        Sentences[i]['embedding'] = sent_to_embed[sent]

    if not data_dir.endswith("/"):
        data_dir += "/"

    file_list = open("USE_Data_files/" + year + \
            "/train_file_list.txt").read().strip().split("\n")

    #Loop over all files again and get all source-target pairs
    print("YEAR: " + year + " : Looping over all files")

    # for root, subdir, files in os.walk(data_dir + year):
        # for filename in files:

    for filename in file_list:
            f = open(filename)
            filename = filename.split("/")[-1]

            try:
                f_pairs = json.load(f)
            except:
                print("Cannot load " + filename)
                continue

            for dct in f_pairs:
                source = dct['source']
                target = dct['target']
                jacana_sim = dct["score"]
                sureAlign = dct["sureAlign"]
                possibleAlign = dct["possibleAlign"]

                try:
                    source_ind = sent_to_ind[source]
                    target_ind =sent_to_ind[target]
                except:
                    print(target)
                    continue

                source_embed = Sentences[source_ind]\
                        ['embedding']
                target_embed = Sentences[target_ind]\
                        ['embedding']

                sim = cos_similarity(source_embed,target_embed)
                ang_sim = angular_similarity(source_embed, \
                                        target_embed)

                Sentences[source_ind]['filename'] = filename
                Sentences[target_ind]['filename'] = filename
                Sentences[source_ind]['type'] = 'summary'
                Sentences[target_ind]['type'] = 'article'

                if 'pairs' not in Sentences[target_ind]:
                    Sentences[target_ind]['pairs'] = {}
                    Sentences[target_ind]['max_score'] = {
                            "score": 0,
                            "summary_sent_ind": -1
                            }

                Sentences[target_ind]['pairs'][source_ind] = \
                            {"cosine_sim": sim,
                             "angular_sim": ang_sim,
                             "jacana_sim": jacana_sim,
                             "sureAlign": sureAlign,
                             "possibleAlign": possibleAlign
                             }
                tmp=Sentences[target_ind]['max_score']['score']

                if sim > tmp:
                    Sentences[target_ind]['max_score'] = {
                            "score": sim,
                            "summary_sent_ind": source_ind
                            }

    ########################################################
    #Save Files
    print(7)
    print("YEAR: " + year + "saving data to " +  "USE_Data_files/" +year + "/Sentences_pairs_dict.pkl")
    print(8)

    try:
        print(9)
        os.system("mv USE_Data_files/" +year + "/Sentences_pairs_dict.pkl " + "USE_Data_files/" + year + "/backup_Sentences_pairs_dict.pkl")
    except:
        print(10)
        pass

    save_dict = {
            "Sentences": Sentences,
            "sent_to_ind": sent_to_ind
            }
    print(11)
    save_file = open("USE_Data_files/" +year + "/Sentences_pairs_dict.pkl", "wb")
    pickle.dump(save_dict, save_file)
    print("YEAR : " + year  +": DONE")


def score_all_year(data_dir, func = score_sentences):
    manager = mp.Queue()
    pool = mp.Pool()
    jobs = []

    for year in os.listdir(data_dir):
        print("YEAR: " + year)
        job = pool.apply_async(func, (data_dir, year ))
        jobs.append(job)

    for job in jobs:
        job.get()

    pool.close()
    pool.join()

def get_statistics(data_dir, year):
    print(year + ": Loading")
    try:
        Sentences = pickle.load(open("./USE_Data_files/" + year + \
            "/Sentences_pairs_dict.pkl", "rb"))
    except Exception as e:
        print(e)

    Sentences = Sentences['Sentences']

    jacana_score = []
    use_cos_score = []
    use_ang_score = []

    print(year + ": Extracting Scores")
    for i in Sentences:
        if "pairs" in Sentences[i]:
            jacana_tmp = []
            ang_sim_tmp = []

            for p in Sentences[i]['pairs']:
                tmp = float(Sentences[i]['pairs'][p]['jacana_sim'])
                jacana_tmp.append(tmp)
                ang_sim_tmp.append( \
                        Sentences[i]['pairs'][p]['angular_sim'] )

            jacana_score.append(max(jacana_tmp))
            use_ang_score.append(max(ang_sim_tmp))
            use_cos_score.append(\
                    Sentences[i]['max_score']['score'] )

    dct = {"jacana": jacana_score,
            "use_cos": use_cos_score,
            "use_ang": use_ang_score}

    print(year + ": Saving files")
    save_file = open("USE_Data_files/" + year + "/statistics.pkl",\
            "wb")
    pickle.dump(dct, save_file)
    print(year + ": DONE")

def get_all_statstics(data_dir):
    manager = mp.Manager()
    pool = mp.Pool()
    jobs = []

    for year in os.listdir("./USE_Data_files/"):
        job = pool.apply_async(get_statistics, (data_dir, year))
        jobs.append(job)

    for job in jobs:
        job.get()

    pool.close()
    pool.join()

def analyze_stats(data_dir, year):
    scores = pickle.load(open("USE_Data_files/" + year + "/statistics.pkl"))
    jac = scores['jacana']
    cos = scores['use_cos']
    ang = scores['use_ang']

    print(year + ":sp:ac-cos:" + str(spearmanr(jac, cos)))
    print(year + ":pr:jac-cos:" + str(pearsonr(jac, cos)))
    # print(year + ":ang-cos:" + str(spearmanr(ang, cos)))
    # print(year + ":jac-ang:" + str(spearmanr(jac, ang)))

def analyze_all_stats(data_dir):
    manager = mp.Manager()
    pool = mp.Pool()
    jobs = []

    for year in os.listdir("./USE_Data_files/"):
        job = pool.apply_async(analyze_stats, (data_dir, year))
        jobs.append(job)

    for job in jobs:
        job.get()

    pool.close()
    pool.join()

#######################################
#######################################

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-news_type", type=str, default="Sports")
    opts = args.parse_args()

    data_dir += opts.news_type + "/"

    print("Creating file lists")

    if not os.path.exists("USE_Data_files"):
        os.mkdir("USE_Data_files")

    for year in os.listdir(data_dir):
        if not os.path.exists("USE_Data_files/"+year):
            os.mkdir("USE_Data_files/" + year)
        get_file_list(data_dir, year)

    # sentences_worker(data_dir, '2002', mp.Manager().Queue())
    print("DONE\n_______________________________________")

    print("Creates Sentences dictionary")
    get_sentences(data_dir)
    print("Done")

    print("Extracting and analyzing sentence pairs")
    score_all_year(data_dir, score_sentences)
    print("DONE\n_______________________________________")

    # score_summaries(data_dir, '1996')

    print("Extracting and analyzing SUMMARY pairs")
    score_all_year(data_dir, score_summaries)
    print("DONE\n_______________________________________")

    get_all_statstics(data_dir)
    analyze_all_stats(data_dir)
