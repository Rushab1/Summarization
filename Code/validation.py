import os
import numpy as np
import sys

def optimize_min_length():
    os.chdir("./Summarizer/OpenNMT-py/")

    if not os.path.exists("TMP"):
        os.mkdir("TMP")

    if not os.path.exists("TMP/by_min_length"):
        os.mkdir("TMP/by_min_length")

    print("Validating - finding optimum length")

    os.system("cp ../../train_test_data/untouched_valid_data/by_sentence_abstracts.txt TMP/by_min_length/orig_abstracts.txt")

    for min_length in range(10, 150, 5):
        print("min_length = " + str(min_length))
        os.system(
            "python translate.py -model ../modelfiles/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt " + 
            "-src ../../train_test_data/untouched_valid_data/by_sentence_articles.txt " +
            "-output TMP/by_min_length/" + str(min_length) + ".txt " + 
            "-verbose -ignore_when_blocking \".\" \"</t>\" \"<t>\"" + 
            " -min_length "  + str(min_length) + " " +
            "-batch_size 1 " + 
            "-gpu 0")

        print("Done")

    os.chdir("../../")

def create_cleaned_by_threshold_files(type_s = "by_sentence"):
    print("Cleaning and creating cleaned_by_sentence_articles.txt files by threshold")

    if not os.path.exists("train_test_data/untouched_valid_data/cleaned_threshold_files"):
        os.mkdir("train_test_data/untouched_valid_data/cleaned_threshold_files")
    if not os.path.exists("train_test_data/untouched_valid_data/cleaned_threshold_files/" + type_s):
        os.mkdir("train_test_data/untouched_valid_data/cleaned_threshold_files/" + type_s)

    # for threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
    for threshold in [0.75]:
        print("threshold = " + str(threshold))
        os.system("python summarize.py -type_s " + type_s + " -threshold " + str(threshold) + " -valid 1 -test 0")

        os.listdir("train_test_data/untouched_valid_data/")
        os.system("cp train_test_data/untouched_valid_data/cleaned_" + type_s + "_articles.txt train_test_data/untouched_valid_data/cleaned_threshold_files/" +type_s +"/"+ str(threshold) +".txt")

def optimize_threshold(min_length = 75, type_s = "by_sentence"):
    os.chdir("./Summarizer/OpenNMT-py/")
    if not os.path.exists("TMP"):
        os.mkdir("TMP")

    if not os.path.exists("TMP/by_threshold"):
        os.mkdir("TMP/by_threshold")

    if not os.path.exists("TMP/by_threshold/" + type_s):
        os.mkdir("TMP/by_threshold/" + type_s)

    os.system("cp ../../train_test_data/untouched_valid_data/" + type_s + "_abstracts.txt TMP/by_threshold/" + type_s + "/orig_abstracts.txt")

    directory = "../../train_test_data/untouched_valid_data/cleaned_threshold_files/" + type_s + "/"

    for f in os.listdir(directory):
        threshold = float(f.replace(".txt", ""))

        cmd  = "python translate.py -model ../modelfiles/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt " 
        cmd += "-src " + directory + f + " "
        cmd += "-output TMP/by_threshold/" + type_s + "/" + f + " " 
        cmd += "-verbose -ignore_when_blocking \".\" \"</t>\" \"<t>\" "
        cmd += "-min_length "  + str(min_length) + " "
        cmd += "-batch_size 2 "
        cmd += "-gpu 0 "

        os.system(cmd)
        print("Done")

    os.chdir("../../")

create_cleaned_by_threshold_files("by_sentence_neg")
# optimize_threshold()
