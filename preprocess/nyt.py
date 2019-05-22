import os
import sys
from shutil import copyfile
from tqdm import tqdm

#Copy all files from all subdirectories of the first directory into a new directory, disregarding the file structure
#WARNING: If two files have same name only one is retained
def all_files_in_one_directory(directory, new_directory):

    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    for root, dirs, files in tqdm(os.walk(directory, topdown=False)):
        for name in files:
            fpath = os.path.join(root, name)
            new_fpath = os.path.join(new_directory, name)
            copyfile(fpath, new_fpath)

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="nyt")
    opts = args.parse_args()

    if opts.dataset == "nyt":
        all_files_in_one_directory("../Data/Datasets/nyt/pair_sent_matched/", "../Data/Datasets/nyt/new_pair_sent_matched/")

