import argparse
import os
import pickle
import numpy as np
import sys
from create_cleaned_files import *
import subprocess
from files2rouge import files2rouge
import multiprocessing as mp

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations", "All"]

def create_validation_files_domain(dataset, domain, type_s, file_list):
    files_dir = os.path.join("../Data/Processed_Data/", dataset, domain, "validation_files")
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    files_dir = os.path.join("../Data/Processed_Data/", dataset, domain, "validation_files", type_s)
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    load_file = os.path.join("../Data/Processed_Data", dataset, domain, type_s, "predictions.pkl")
    print(load_file)
    predictions_dct = pickle.load(open(load_file, "rb"))

    for threshold in [0, 0.3, 0.5, 0.55, 0.6, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.75, 0.8, 0.9]:
        print("Creating validation files for Dataset: {}, Domain: {} and Threshold: {}".format(dataset, domain, str(threshold)))
        save_dir = os.path.join(files_dir, str(threshold))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        create_cleaned_files_list(dataset, file_list, predictions_dct, save_dir, threshold=threshold)

def create_validation_files(dataset, type_s):
    for domain in DOMAINS:
        file_list_file = os.path.join("../Data/Processed_Data/", dataset, domain, "val_file_list.txt")
        file_list = open(file_list_file).read().strip().split("\n")
        create_validation_files_domain(dataset, domain, type_s, file_list)

def opennmt_summarizer(orig_file, cleaned_file, output_dir, min_length, orig = False):
    os.chdir("../Summarizers/OpenNMT-py/")
    cmd = ['python', 'translate.py',
            '-model', '../../modelfiles/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt',
            '-src', cleaned_file,
            '-output', os.path.join(output_dir, 'pred_cleaned,.txt'),
            '-ignore_when_blocking', '"." "</t>" "<t>"',
            '-min_length', str(min_length),
            '-batch_size', str(3),
            '-gpu', '0']
    subprocess.call(cmd)

    if not orig:
        os.chdir("../../Code")
        return

    cmd = ['python', 'translate.py',
            '-model', '../../modelfiles/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt',
            '-src', orig_file,
            '-output', os.path.join(output_dir, 'pred_orig,.txt'),
            '-ignore_when_blocking', '"." "</t>" "<t>"',
            '-min_length', str(min_length),
            '-batch_size', str(3),
            '-gpu', '0']

    subprocess.call(cmd)
    os.chdir("../../Code")

def fast_abs_summarizer(orig_file, cleaned_file, output_dir, min_length, orig = False):
    classpath = os.path.abspath("../packages/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar")
    os.environ["CLASSPATH"] = classpath
    output_dir = os.path.abspath(output_dir)
    cmd = ['python', 'summarize.py',
            '-articles_file', os.path.abspath(cleaned_file),
            '-abstracts_file', os.path.join(output_dir, 'abstracts.txt'),
            '-output_file', os.path.join(output_dir, 'pred_cleaned.txt'),
            '-min_length', str(min_length),
            '-batchsize', str(1000),
            ]

    os.chdir("../Summarizers/fast_abs_rl/")
    print("\n\n" + " ".join(cmd) + "\n\n")
    subprocess.call(cmd)
    os.chdir("../../Code")

    if not orig:
        return

    cmd = ['python', 'summarize.py',
            '-articles_file', os.path.abspath(orig_file),
            '-abstracts_file', os.path.join(output_dir, 'abstracts.txt'),
            '-output_file', os.path.join(output_dir, 'pred_orig.txt'),
            '-min_length', str(min_length),
            '-batchsize', str(1000)
            ]


    os.chdir("../Summarizers/fast_abs_rl/")
    subprocess.call(cmd)
    os.chdir("../../Code")

def validate_domain(dataset, domain, type_s, summarizer):
    val_dir = os.path.join("../Data/Processed_Data/", dataset, domain, "validation_files", type_s)
    val_dir = os.path.abspath(val_dir)

    subdirs = os.listdir(val_dir)
    subdirs = ['0.66', '0.67', '0.68', '0.7' ]
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

        if summarizer.lower() == "opennmt" or summarizer.lower() == "opennmt-py":
            opennmt_summarizer(orig_file, cleaned_file, threshold_dir, min_length = 75, orig = orig)

        print("Validating for threshold = " + threshold_dir.split("/")[-1])
        if summarizer.lower() == "fastabs" or summarizer.lower() == "fast-abs":
            fast_abs_summarizer(orig_file, cleaned_file, threshold_dir, min_length = 75, orig = orig)

def calculate_rouge_parallel(val_dir, threshold_dir, JobQueue):
    threshold_dir = os.path.abspath(os.path.join(val_dir, threshold_dir))
    abstracts_file = os.path.join(threshold_dir, "abstracts.txt")
    pred_cleaned_file = os.path.join(threshold_dir, "pred_cleaned.txt")

    f = open(abstracts_file).read().replace("<t>", "")
    f = f.replace("</t>", "")
    g = open(abstracts_file, "w")
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
    cleaned_text = cleaned_text.replace("<t>", "")
    cleaned_text = cleaned_text.replace("</t>", "")

    orig_articles = orig_text.strip().split("\n")
    cleaned_articles = cleaned_text.strip().split("\n")
    avg_reduction = 0

    if len(orig_articles) != len(cleaned_articles):
        JobQueue.put(None)
        return

    for i in range(0, len(orig_articles)):
        clean_len = len(cleaned_articles[i].split())
        orig_len  = len(orig_articles[i].split())
        tmp = 1 - 1.0 * clean_len / orig_len
        avg_reduction += 1.0 / len(orig_articles) * tmp
    JobQueue.put( ( threshold, f, avg_reduction ) )

def calculate_rouge(dataset, domain, type_s, save_file = "./rouge.pkl"):
    val_dir = os.path.join("../Data/Processed_Data/", dataset, domain, "validation_files", type_s)
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
        job = pool.apply_async( calculate_rouge_parallel,
                                (val_dir, threshold_dir, JobQueue)
                                )
        jobs.append(job)

    for job in jobs:
        job.get()

    JobQueue.put("kill")

    res = JobQueue.get()
    while res != "kill":
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

        print("=============================")
        print(threshold, avg_reduction, f["rouge-l"]["average_f"])
        print("=============================")
        res = JobQueue.get()

    pool.close()
    pool.join()

    pickle.dump(sorted(zip(th, redn, r1,r2,rl,), key=lambda x:x[0]), open(save_file,  "wb"))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default=None)
    args.add_argument("-type_s", type=str, default = "importance")
    args.add_argument("-summarizer", type=str, default = "opennmt")
    opts = args.parse_args()

    if opts.dataset in ["cnn", "cnndm", "gigaword"] :
        DOMAINS = ['All']

    DOMAINS = ['All']
    # create_validation_files(opts.dataset, opts.type_s)
    for domain in DOMAINS:
        validate_domain(opts.dataset, domain, opts.type_s, opts.summarizer)

    calculate_rouge(opts.dataset, "All", "importance", opts.dataset + "_" + opts.type_s + "_" + opts.summarizer + "_rouge.pkl")
