import argparse
import os
import pickle
import numpy as np
import sys
from create_cleaned_files import *
import subprocess

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

    for threshold in [0, 0.15, 0.3, 0.45, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85]:
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


def opennmt_summarizer(orig_file, cleaned_file, output_dir, min_length):
    os.chdir("../Summarizers/OpenNMT-py/")
    cmd = ['python', 'translate.py',
            '-model', '../../modelfiles/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt',
            '-src', orig_file,
            '-output', os.path.join(output_dir, 'pred_orig,.txt'),
            '-ignore_when_blocking', '"." "</t>" "<t>"',
            '-min_length', str(min_length),
            '-batch_size', str(2),
            '-gpu', '0']
    subprocess.call(cmd)

    cmd = ['python', 'translate.py',
            '-model', '../../modelfiles/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt',
            '-src', cleaned_file,
            '-output', os.path.join(output_dir, 'pred_cleaned,.txt'),
            '-ignore_when_blocking', '"." "</t>" "<t>"',
            '-min_length', str(min_length),
            '-batch_size', str(2),
            '-gpu', '0']
    subprocess.call(cmd)
    os.chdir("../../Code")

def validate_domain(dataset, domain, type_s, summarizer):
    val_dir = os.path.join("../Data/Processed_Data/", dataset, domain, "validation_files", type_s)
    val_dir = os.path.abspath(val_dir)

    for threshold_dir in os.listdir(val_dir):
        threshold_dir = os.path.abspath(os.path.join(val_dir, threshold_dir))
        orig_file = os.path.join(threshold_dir, "orig.txt")
        cleaned_file = os.path.join(threshold_dir, "cleaned.txt")
        output_dir = threshold_dir

        if summarizer.lower() == "opennmt" or summarizer.lower() == "opennmt-py":
            opennmt_summarizer(orig_file, cleaned_file, threshold_dir, min_length = 75)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default=None)
    args.add_argument("-type_s", type=str, default = "importance")
    args.add_argument("-summarizer", type=str, default = "opennmt")
    opts = args.parse_args()


    if opts.dataset == "cnn":
        DOMAINS = ['All']

    create_validation_files(opts.dataset, opts.type_s)
    for domain in DOMAINS:
        validate_domain(opts.dataset, domain, opts.type_s, opts.summarizer)

