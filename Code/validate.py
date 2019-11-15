import argparse
import os
import pickle
import numpy as np
import sys
from create_cleaned_files import *
import subprocess
from files2rouge import files2rouge
import multiprocessing as mp
from IPython import embed
import time
import random

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations", "All"]

# THRESHOLDS = [0, 0.55, 0.6, 0.65, 0.67, 0.7, 0.75, 0.8] #reduced CNNDM
# THRESHOLDS = [ 0.65, 0.67, 0.7, 0.75, 0.8] #reduced CNNDM
# THRESHOLDS = [0, 0.3, 0.5, 0.55, 0.6, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.75, 0.8, 0.9] # For cnndm

THRESHOLDS = [0, 0.5, 0.6, 0.61, 0.62, 0.63, 0.64 , 0.65, 0.55, 0.575, 0.625, 0.675,  0.7, 0.8 ]  # For nytimes

MIN_LENGTH = 75
NUM_NYT_TEST_FILES = 10000

for i in range(0, len(THRESHOLDS)):
    THRESHOLDS[i] = str(THRESHOLDS[i])

def create_validation_files_domain(dataset, domain, type_s, file_list, split = "validation"):
    files_dir = os.path.join("../Data/Processed_Data/", dataset, domain, split + "_files")
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    files_dir = os.path.join("../Data/Processed_Data/", dataset, domain, split + "_files", type_s)
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    load_file = os.path.join("../Data/Processed_Data", dataset, domain, type_s, "predictions.pkl")
    print(load_file)
    predictions_dct = pickle.load(open(load_file, "rb"))

    # for threshold in [0, 0.3, 0.5, 0.55, 0.6, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.75, 0.8, 0.9]: # For cnndm
    # for threshold in [0, 0.5, 0.55, 0.575, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65,  0.8 ]:  # For nytimes
    for threshold in THRESHOLDS:
        threshold = float(threshold)
        print("Creating " + split + " files for Dataset: {}, Domain: {} and Threshold: {}".format(dataset, domain, str(threshold)))
        save_dir = os.path.join(files_dir, str(threshold))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        create_cleaned_files_list(dataset, file_list, predictions_dct, save_dir, threshold=threshold)

def create_validation_files(dataset, type_s, split = "validation"):
    for domain in DOMAINS:
        if split == "validation":
            file_list_file = os.path.join("../Data/Processed_Data/", dataset, domain, "val_file_list.txt")
        if split == "test":
            file_list_file = os.path.join("../Data/Processed_Data/", dataset, domain, "test_file_list.txt")

        file_list = open(file_list_file).read().strip().split("\n")
        if dataset == "nyt":
            import random
            file_list = random.sample(file_list, NUM_NYT_TEST_FILES)

        create_validation_files_domain(dataset, domain, type_s, file_list, split)

def lead_summarizer(orig_file, cleaned_file, output_dir, min_length, orig = False):
    articles = open(cleaned_file).read().split("\n")
    pred_summaries = []
    for article in articles:
        pred_summaries.append( " ".join (article.split()[:min_length] ))

    output_file = os.path.join(output_dir,'pred_cleaned_lead' + str(min_length) + '.txt')
    print(output_file)
    f = open(output_file, "w")
    f.write("\n".join(pred_summaries))
    f.close()

def opennmt_summarizer(orig_file, cleaned_file, output_dir, min_length, orig = False):
    files = [orig_file, cleaned_file]
    for j in range(0, 2):
        fname = files[j]
        f = open(fname).read().split("\n")
        for i in range(0,len(f)):
            f_spl = f[i].split()

            if len(f_spl) > 5000:
                f[i] = " ".join(f_spl[:5000])

            if len(f_spl) < 10:
                f[i] = "This article has been replaced because it was empty. This has almost no effect on the Rouge scores."

        r = str(int(random.random() * 10**10))
        g = open("../TMP/" + r + ".txt", "w")
        g.write("\n".join(f))
        g.close()
        files[j] = os.path.abspath("../TMP/" + r + ".txt")
        print(r)
    orig_file = files[0]
    cleaned_file = files[1]

    os.chdir("../Summarizers/OpenNMT-py/")
    cmd = ['python', 'translate.py',
            '-model', '../../modelfiles/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt',
            '-src', cleaned_file,
            '-output', os.path.join(output_dir,
                       'pred_cleaned_opennmt' + str(min_length) + '.txt'),
            '-ignore_when_blocking', '"." "</t>" "<t>"',
            '-min_length', str(min_length),
            '-batch_size', str(2),
            '-gpu', '0',
            ]

    print(" ".join(cmd))
    start_time = time.time()
    subprocess.call(cmd)
    run_time = time.time() - start_time

    if not orig:
        os.chdir("../../Code")
        return run_time

    # cmd = ['python', 'translate.py',
            # '-model', '../../modelfiles/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt',
            # '-src', orig_file,
            # '-output', os.path.join(output_dir, 'pred_orig_opennmt' + str(min_length) + '.txt'),
            # '-ignore_when_blocking', '"." "</t>" "<t>"',
            # '-min_length', str(min_length),
            # '-batch_size', str(1),
            # '-gpu', '0',
            # ]

    # subprocess.call(cmd)
    os.chdir("../../Code")
    return run_time

def fast_abs_summarizer(orig_file, cleaned_file, output_dir, min_length, orig = False):
    classpath = os.path.abspath("../packages/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar")
    os.environ["CLASSPATH"] = classpath
    output_dir = os.path.abspath(output_dir)
    cmd = ['python', 'summarize.py',
            '-articles_file', os.path.abspath(cleaned_file),
            '-abstracts_file', os.path.join(output_dir, 'abstracts.txt'),
            '-output_file', os.path.join(output_dir, 'pred_cleaned_fastabs' + str(min_length) +'.txt'),
            '-min_length', str(min_length),
            '-batchsize', str(500),
            ]

    os.chdir("../Summarizers/fast_abs_rl/")
    print("\n\n" + " ".join(cmd) + "\n\n")
    subprocess.call(cmd)
    os.chdir("../../Code")

    if not orig:
        return

    # cmd = ['python', 'summarize.py',
            # '-articles_file', os.path.abspath(orig_file),
            # '-abstracts_file', os.path.join(output_dir, 'abstracts.txt'),
            # '-output_file', os.path.join(output_dir, 'pred_orig_opennmt' + str(min_length) + '.txt'),
            # '-min_length', str(min_length),
            # '-batchsize', str(100)
            # ]


    # os.chdir("../Summarizers/fast_abs_rl/")
    # subprocess.call(cmd)
    # os.chdir("../../Code")

def validate_domain(dataset, domain, type_s, summarizer, split = "validation"):
    val_dir = os.path.join("../Data/Processed_Data/", dataset, domain, split + "_files", type_s)
    val_dir = os.path.abspath(val_dir)

    subdirs = os.listdir(val_dir)
    subdirs = THRESHOLDS
    for threshold_dir in subdirs:
        threshold_dir = os.path.abspath(os.path.join(val_dir, threshold_dir))
        orig_file = os.path.join(threshold_dir, "orig.txt")
        cleaned_file = os.path.join(threshold_dir, "cleaned.txt")
        output_dir = threshold_dir

        threshold = float(threshold_dir.split("/")[-1])
        if threshold == 0:
            orig = True
        else:
            orig = False

        print(split + " : threshold = " + threshold_dir.split("/")[-1])

        if summarizer.lower() == "lead":
            lead_summarizer(orig_file, cleaned_file, threshold_dir, min_length = MIN_LENGTH, orig = orig)
        if summarizer.lower() == "opennmt" or summarizer.lower() == "opennmt-py":
            run_time = opennmt_summarizer(orig_file, cleaned_file, threshold_dir, min_length = MIN_LENGTH, orig = orig)
            print(threshold, "Time: " + str(run_time))

        if summarizer.lower() == "fastabs" or summarizer.lower() == "fast-abs":
            fast_abs_summarizer(orig_file, cleaned_file, threshold_dir, min_length = MIN_LENGTH, orig = orig)

def calculate_rouge_parallel(val_dir, threshold_dir, summarizer, JobQueue):
    threshold_dir = os.path.abspath(os.path.join(val_dir, threshold_dir))
    abstracts_file = os.path.join(threshold_dir, "abstracts.txt")
    pred_cleaned_file = os.path.join(threshold_dir, "pred_cleaned_" + summarizer + str(MIN_LENGTH) + ".txt")

    for fname in [abstracts_file, pred_cleaned_file]:
        f = open(fname).read()
        f = f.replace("<t>", "")
        f = f.replace("</t>", "")
        g = open(fname, "w")
        g.write(f)
        g.close()
    threshold = float(threshold_dir.split("/")[-1])

    print(threshold)
    try:
        f = files2rouge.run(pred_cleaned_file, abstracts_file)
    except Exception as e:
        print(e)
        JobQueue.put( None )
        return

    #calculate reduction after cleaning
    orig_text = open(os.path.join(threshold_dir, "orig.txt")).read()
    cleaned_text = open(os.path.join(threshold_dir, "cleaned.txt")).read()

    orig_text = orig_text.replace("<t>", "")
    orig_text = orig_text.replace("</t>", "")

    cleaned_text = re.sub(  "<t> [  ]*?</t>","NO_TEXT_IN_CLEANED_ARTICLE", cleaned_text)
    tmp = cleaned_text.split("\n")
    i = 0
    while tmp[i].strip() == "" or i >= len(tmp):
        tmp [i] = "NO_TEXT_IN_CLEANED_ARTICLE"
        i = i + 1

    cleaned_articles = "\n".join(tmp)

    cleaned_text = cleaned_text.replace("<t>", "")
    cleaned_text = cleaned_text.replace("</t>", "")

    # orig_articles = orig_text.strip().split("\n")
    # cleaned_articles = cleaned_text.strip().split("\n")
    orig_articles = orig_text.split("\n")
    cleaned_articles = cleaned_text.split("\n")
    avg_reduction = 0

    if len(orig_articles) != len(cleaned_articles):
        print("NONE: 1")
        JobQueue.put(None)
        return

    for i in range(0, len(orig_articles)):
        clean_len = len(cleaned_articles[i].split())
        orig_len  = len(orig_articles[i].split())
        try:
            tmp = 1 - 1.0 * clean_len / orig_len
        except:
            tmp = 0
        avg_reduction += 1.0 / len(orig_articles) * tmp
    JobQueue.put( ( threshold, f, avg_reduction ) )

def calculate_rouge(dataset, domain, type_s, summarizer, split = "validation", save_file = "./rouge.pkl"):
    val_dir = os.path.join("../Data/Processed_Data/", dataset, domain, split + "_files", type_s)
    val_dir = os.path.abspath(val_dir)

    r1 = []
    r2 = []
    rl = []
    th = []
    redn = []

    import random
    subdirs = os.listdir(val_dir)
    random.shuffle(subdirs)

    manager = mp.Manager()
    pool = mp.Pool()
    JobQueue = manager.Queue()
    jobs = []

    for threshold_dir in subdirs:
        pred_fname = "pred_cleaned_" + summarizer + str(MIN_LENGTH) + ".txt"
        if not os.path.exists(os.path.join(val_dir, threshold_dir, pred_fname )):
            print("ERROR 1 ", os.path.join(val_dir, threshold_dir, pred_fname) )
            continue


        job = pool.apply_async( calculate_rouge_parallel,
                                (val_dir, threshold_dir, summarizer, JobQueue)
                                )
        jobs.append(job)

    for job in jobs:
        job.get()

    JobQueue.put("kill")

    res = JobQueue.get()
    while res != "kill":
        print("RES========== " + str(res))
        if res == None:
            res = JobQueue.get()
            continue
        threshold, f, avg_reduction = res

        #Save Rouge values
        r1.append(f["rouge-1"]["average_f"])
        r2.append(f["rouge-2"]["average_f"])
        rl.append(f["rouge-l"]["average_f"])
        th.append(threshold)
        redn.append(avg_reduction)
        res = JobQueue.get()

    pool.close()
    pool.join()

    pickle.dump(sorted(zip(th, redn, r1,r2,rl,), key=lambda x:x[0]), open(save_file,  "wb"))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default=None)
    args.add_argument("-type_s", type=str, default = "importance")
    args.add_argument("-summarizer", type=str, default = "opennmt")
    args.add_argument("-split", type=str, default = "validation")
    args.add_argument("-min_length", type=int, default = 75)
    opts = args.parse_args()

    MIN_LENGTH = opts.min_length
    if opts.dataset in ["cnn", "cnndm", "gigaword"] :
        DOMAINS = ['All']

    DOMAINS = ['All']

    # create_validation_files(opts.dataset, opts.type_s, opts.split)

    for domain in DOMAINS:
        validate_domain(opts.dataset, domain, opts.type_s, opts.summarizer, opts.split)

    save_file = "Results/" + opts.split.upper() + "_" + opts.dataset + "_" + opts.type_s + "_" + opts.summarizer + str(MIN_LENGTH) + "_rouge.pkl"
    calculate_rouge(opts.dataset, "All", opts.type_s, opts.summarizer, opts.split, save_file)
