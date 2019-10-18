import pickle
from create_cleaned_files import *
import argparse
import random

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_random_file(opts):
    orig = open(os.path.join(save_dir, "orig.txt"), "r").read()
    cleaned = open(os.path.join(save_dir, "cleaned.txt"), "r").read()
    abstracts = open(os.path.join(save_dir, "abstracts.txt"), "r").read()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default="nyt")
    args.add_argument("-type_s", type=str, default=None) #Do not specify if you want both classifiers
    args.add_argument("-split", type=str, default="valid") #valid/test
    args.add_argument("-domain", type=str, default=None) #Business/Sports/Science/USIntlRelations/All=None
    args.add_argument("-save_dir", type=str, default="../TMP/") #Business/Sports/Science/USIntlRelations/All=None
    args.add_argument("-threshold", type=float, default = 0.7) #Business/Sports/Science/USIntlRelations/All=None
    args.add_argument("-classifier_dataset", type=str, default="nyt")
    args.add_argument("-t", type=int, default = 0)
    # args.add_argument("-classifier_type_s", type=str, default="importance") #Do not specify if you want both classifiers
    # args.add_argument("-classifier_domain", type=str, default=None) #Business/Sports/Science/USIntlRelations/All=None
    opts = args.parse_args()

    if opts.split == "valid":
        opts.split = "val"

    if opts.domain == None:
        opts.domain = "All"

    # if opts.classifier_domain == None:
        # opts.classifier_domain = "All"

    dataset = opts.dataset
    split = opts.split
    domain = opts.domain
    type_s = opts.type_s
    save_dir = opts.save_dir
    threshold = opts.threshold
    classifier_dataset = opts.classifier_dataset
    # classifier_domain = opts.classifier_domain
    # classifier_type_s = opts.classifier_type_s

    orig        = open(os.path.join( "../Data/Processed_Data", dataset, domain,  "validation_files", type_s, str(threshold), "orig.txt")).read().split("\n")
    cleaned     = open(os.path.join( "../Data/Processed_Data", dataset, domain,  "validation_files", type_s, str(threshold), "cleaned.txt")).read().split("\n")
    abstracts   = open(os.path.join( "../Data/Processed_Data", dataset, domain,  "validation_files", type_s, str(threshold), "abstracts.txt")).read().split("\n")
    pred        = open(os.path.join( "../Data/Processed_Data", dataset, domain,  "validation_files", type_s, str(threshold), "pred_cleaned.txt")).read().split("\n")

    n = len(orig)
    t = random.randint(0, n)
    t = opts.t

    o = orig[t]
    c = cleaned[t]
    a = abstracts[t]
    p = pred[t]

    import re
    pattern = re.compile("<t>(.*?)</t>")
    o = re.findall(pattern, o)
    c = re.findall(pattern, c)

    
    if classifier_dataset != dataset:
        predictions = pickle.load(open("../Data/Cross_Classifier_Predictions/" + classifier_dataset + "/" + dataset + ".pkl", "rb" ))

    print(threshold)
    print("\n###### ORIG ABSTRACT      ###########################")
    print(a)

    if classifier_dataset == dataset and type_s == "importance":
        print("\n###### PREDICTED ABSTRACT ###########################")
        print(p)

    print("\n######       ARTICLE      ###########################")
    if classifier_dataset != dataset:
        for s in o:
            score = predictions[s.strip()]
            if score[1] > threshold:
                print(bcolors.WARNING + s.strip() + bcolors.ENDC)
            else:
                print(bcolors.FAIL + s.strip() + bcolors.ENDC)
    else:
     for s in o:
        if s in c:
            print(bcolors.WARNING + s.strip() + bcolors.ENDC)
        else:
            print(bcolors.FAIL + s.strip() + bcolors.ENDC)
