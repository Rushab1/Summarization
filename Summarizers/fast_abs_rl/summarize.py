import os
import sys
import argparse
from IPython import embed

categories = {"Business": 0, "Sports": 1, "Science":2, "Politics":3, "All":-1}
unknown_article_string = "<unknown> This article has been completely cleaned. Just adding this text so that summarizer doesn't crash. We need to have at least max_dec_words number of words so we will be repeating whatever we said till now again.  This article has been completely cleaned. Just adding this text so that summarizer doesn't crash. We need to have at least max_dec_words number of words so we will be repeating whatever we said till now again ."
MODEL_DIR = os.path.abspath("../../modelfiles/chen_and_bansal/new/")

def test_summaries(articles_file, abstracts_file, output_file, min_length, batchsize=1000):
    if not os.path.exists("preprocess/TMP"):
        os.mkdir("preprocess/TMP")

    f = open(articles_file).read().strip()

    f = f.split("\n")
    for i in range(0, len(f)):
        if len (f[i]) < 100:
            f[i] += unknown_article_string

    # f = "\n".join(f)
    # f = f.replace("\n\n", "\n" + unknown_article_string + "\n")
    # f = f.split("\n")

    print("Copying " + str(len(f)) + " articles to preprocess/TMP/src_orig.txt")
    g = open("./preprocess/TMP/src_orig.txt", "w")
    g.write("\n".join(f))
    g.close()

    f = open(abstracts_file).read().strip()
    f = f.split("\n")
    print("Copying " + str(len(f)) + " abstracts to preprocess/TMP/abstracts.txt")
    g = open("./preprocess/TMP/abstracts.txt", "w")
    g.write("\n".join(f))
    g.close()

    #Clear previous outputs
    if not os.path.exists("preprocess/TMP/finished_files"):
        os.mkdir("preprocess/TMP/finished_files")
    os.system("rm -rf preprocess/TMP/finished_files/*")
    os.system("rm -rf preprocess/TMP/decoded_files/*")

    ######################
    #######  SUMMARIZE #######
    ######################
    print("Preprocessing")
    os.chdir("./preprocess")
    cmd = "python md.py --articles_file=TMP/src_orig.txt --abstracts_file ./TMP/abstracts.txt --output_dir TMP/finished_files/test"
    os.system(cmd)
    os.chdir("..")
    print("Done")

    print("\nDecoding ... " )
    os.environ["DATA"] = "preprocess/TMP/finished_files/"

    cmd = "python3 decode_full_model.py --path=preprocess/TMP/decoded_files --model_dir=." + MODEL_DIR + " --beam=1 --test --batch=" + str(batchsize) + " --save_file " + output_file

    print(cmd)
    os.system(cmd)
    print("Done")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-articles_file", type=str)
    args.add_argument("-abstracts_file", type=str)
    args.add_argument("-output_file", type=str)
    args.add_argument("-min_length", type=int)
    args.add_argument("-batchsize", type=int, default=1000)
    opts = args.parse_args()

    test_summaries(opts.articles_file, opts.abstracts_file, opts.output_file, opts.min_length, opts.batchsize)

    # test_summaries("All", "by_sentence", 75, 0.625)
    # test_summaries("Business", "by_sentence", 75, 0.625)
    # test_summaries("Sports", "by_sentence", 25, 0.65)
    # test_summaries("Science", "by_sentence", 75, 0.65)
    # test_summaries("Politics", "by_sentence", 75, 0.7)

    # test_summaries("All", "by_sentence_neg", 75, 0.75)
    # test_summaries("Business", "by_sentence_neg", 75, 0.7)
    # test_summaries("Sports", "by_sentence_neg", 25, 0.75)
    # test_summaries("Science", "by_sentence_neg", 75, 0.75)
    #test_summaries("Politics", "by_sentence_neg", 75, 0.8)
