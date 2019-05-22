import os

categories = {"Business": 0, "Sports": 1, "Science":2, "Politics":3, "All":-1}

unknown_article_string = "<unknown> This article has been completely cleaned. Just adding this text so that summarizer doesn't crash. We need to have at least max_dec_words number of words so we will be repeating whatever we said till now again.  This article has been completely cleaned. Just adding this text so that summarizer doesn't crash. We need to have at least max_dec_words number of words so we will be repeating whatever we said till now again ."

def test_summaries(category, type_s, min_length, threshold, h = 1000, batchsize=1000):
    if not os.path.exists("./Data/nyt_output"):
        os.mkdir("./Data/nyt_output")

    if categories[category] != -1:
        s = categories[category] * h
        e = (1 + categories[category]) * h
    else:
        s = 0
        e = h * 4

    folder =  "Data/final_test_data/untouched_test_data/" + type_s + "/" + str(threshold) + "/"

    f = open(folder + type_s + "_articles.txt").read().strip()
    f = f.replace("\n\n", "\n" + unknown_article_string + "\n")
    f = f.split("\n")[s:e]

    print("Copying " + str(len(f)) + " uncleaned articles to preprocess/TMP/src_orig.txt")
    g = open("./preprocess/TMP/src_orig.txt", "w")
    g.write("\n".join(f))
    g.close()

    f = open(folder + type_s + "_abstracts.txt").read().strip().split("\n")[s:e]
    print("Copying " + str(len(f)) + " abstracts to preprocess/TMP/abstracts.txt")
    g = open("./preprocess/TMP/abstracts.txt", "w")
    g.write("\n".join(f))
    g.close()

    f = open(folder + "cleaned_" + type_s + "_articles.txt").read().strip()
    f = f.replace("\n\n", "\n" + unknown_article_string + "\n")
    f = f.split("\n")[s:e]

    print("Copying " + str(len(f)) + " cleaned articles to preprocess/TMP/src_cleaned.txt")
    g = open("./preprocess/TMP/src_cleaned.txt", "w")
    g.write("\n".join(f))
    g.close()

    #Clear previous outputs
    if not os.path.exists("preprocess/TMP/finished_files"):
        os.mkdir("preprocess/TMP/finished_files")
    os.system("rm -rf preprocess/TMP/finished_files/*")
    os.system("rm -rf preprocess/TMP/decoded_files/*")

    ######################
    #######  ORIG  #######
    ######################
    print("Preprocessing")
    os.chdir("./preprocess")
    cmd = "python md.py --articles_file=TMP/src_orig.txt --abstracts_file ./TMP/abstracts.txt --output_dir TMP/finished_files/test"
    os.system(cmd)
    os.chdir("..")
    print("Done")

    print("\nDecoding: " + str((category,type_s,min_length,threshold,h)))
    fname = category + "_" + str(min_length) + "_" + str(threshold) + "_" + type_s + "_orig.txt"

    os.environ["DATA"] = "preprocess/TMP/finished_files/"

    cmd = "python3 decode_full_model.py --path=preprocess/TMP/decoded_files --model_dir=./pretrained/new/ --beam=1 --test --batch=" + str(batchsize) + " --save_file ./Data/nyt_output/" + fname

    print(cmd)
    os.system(cmd)

    ######################
    #######  CLEANED  #######
    ######################
    os.system("rm -rf preprocess/TMP/finished_files/*")
    os.system("rm -rf preprocess/TMP/decoded_files/*")

    print("\n\nPreprocessing cleaned articles")
    os.chdir("./preprocess")
    cmd = "python md.py --articles_file=TMP/src_cleaned.txt --abstracts_file TMP/abstracts.txt --output_dir TMP/finished_files/test"
    os.system(cmd)
    os.chdir("..")
    print("Done")

    print("\nDecoding: " + str((category,type_s,min_length,threshold,h)))
    fname = category + "_" + str(min_length) + "_" + str(threshold) + "_" + type_s + "_cleaned.txt"

    os.environ["DATA"] = "preprocess/TMP/finished_files/"

    cmd = "python3 decode_full_model.py --path=preprocess/TMP/decoded_files --model_dir=./pretrained/new/ --beam=1 --test --batch=" + str(batchsize) + " --save_file ./Data/nyt_output/" + fname

    print(cmd)
    os.system(cmd)

    print("Done")

    ######################
    #######  ABSTRACTS  #######
    ######################
    fname = category + "_" + str(min_length) + "_" + str(threshold) + "_" + type_s + "_abstracts.txt"
    cmd = "cp preprocess/TMP/abstracts.txt Data/nyt_output/" + fname
    os.system(cmd)


def validate_by_threshold(type_s, batchsize=100):
    flist = os.listdir("Data/cleaned_threshold_files/" + type_s)
    input_dir = "Data/cleaned_threshold_files/" + type_s + "/"

    for fname in flist:
        if fname in os.listdir("Data/cleaned_threshold_files/output/" + type_s):
            print("______________________________________________")
            print(fname + "CONTINUINg")
            print("______________________________________________")
            continue

        #Clean previous outputs
        if not os.path.exists("preprocess/TMP/finished_files"):
            os.mkdir("preprocess/TMP/finished_files")
        os.system("rm -rf preprocess/TMP/finished_files/*")
        os.system("rm -rf preprocess/TMP/decoded_files/*")

        #unknown token handling
        f = open(input_dir + fname)
        s = f.read()
        f.close()
        s = s.replace("\n\n", "\n" + unknown_article_string + "\n")
        s = s.replace("\n<unknown>\n", "\n" + unknown_article_string + "\n")


        s = s.split("\n")

        for i in range(0, len(s)):
            if len(s[i].split()) < 30:
                s[i] = s[i]  + " " + s[i] + " " + s[i]
            if len(s[i].split()) < 30:
                s[i] = s[i]  + " " + s[i] + " " + s[i]

        s = "\n".join(s)
        g = open(input_dir +  fname, "w")
        g.write(s)
        g.close()


        #Preprocess
        print("Preprocessing")
        os.chdir("./preprocess")
        cmd = "python md.py --articles_file=../" + input_dir + fname + " --abstracts_file ../Data/cleaned_threshold_files/abstracts.txt --output_dir TMP/finished_files/test"
        os.system(cmd)
        os.chdir("..")
        print("Done")

        #Decoding
        print("\nDecoding: ")

        os.environ["DATA"] = "preprocess/TMP/finished_files/"

        cmd = "python3 decode_full_model.py --path=preprocess/TMP/decoded_files --model_dir=./pretrained/new/ --beam=1 --test --batch=" + str(batchsize) + " --save_file ./Data/cleaned_threshold_files/output/" + type_s + "/" + fname

        print(cmd)
        os.system(cmd)



if __name__ == "__main__":
    validate_by_threshold("by_sentence")
    validate_by_threshold("by_sentence_neg")
    #test_summaries("All", "by_sentence", 75, 0.625)
    #test_summaries("Business", "by_sentence", 75, 0.625)
    #test_summaries("Sports", "by_sentence", 25, 0.65)
    #test_summaries("Science", "by_sentence", 75, 0.65)
    #test_summaries("Politics", "by_sentence", 75, 0.7)

    #test_summaries("All", "by_sentence_neg", 75, 0.75)
    #test_summaries("Business", "by_sentence_neg", 75, 0.7)
    #test_summaries("Sports", "by_sentence_neg", 25, 0.75)
    #test_summaries("Science", "by_sentence_neg", 75, 0.75)
    #test_summaries("Politics", "by_sentence_neg", 75, 0.8)
