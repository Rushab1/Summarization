#Gathers training and test data from Business, Sports, Politics and Science
import os
import pickle
import sys
import random

categories = ["Business", "Sports", "Science", "Politics"]

def shuffle(x1, x2):
    x = zip(x1, x2)
    random.shuffle(x)
    x1, x2 = zip(*x)
    return [list(x1), list(x2)]

def gather_data(type_s):
    train_data = [[], []]
    test_data = [[], []]
    untouched_test_data = []

    if not os.path.exists("./train_test_data/" + type_s  + "/"):
        os.mkdir("train_test_data/" + type_s + "/")

    os.system("rm -rf train_test_data/" + type_s + "/*")
    
    print("Gathering training data: " + type_s)
    for category in categories:
        print(category)
        category_train_data = pickle.load(open("../" + category + "_NYT/train_test_data/" + type_s + "/train.pkl"))
        train_data[0].extend(category_train_data[0])
        train_data[1].extend(category_train_data[1])

        category_test_data = pickle.load(open("../" + category + "_NYT/train_test_data/" + type_s + "/test.pkl"))
        test_data[0].extend(category_test_data[0])
        test_data[1].extend(category_test_data[1])
        print("done")

    train_data = shuffle(train_data[0], train_data[1])
    test_data = shuffle(test_data[0], test_data[1])

    print("saving")
    pickle.dump(train_data, open("train_test_data/" + type_s + "/train.pkl", "wb"))
    pickle.dump(test_data, open("train_test_data/" + type_s + "/test.pkl", "wb"))
    print("done")


#first 1000 articles from busines, next from sports, science, politics resp.
def gather_untouched_test_data(type_s, threshold = 5000, num_test_articles = 1000, valid = False, num_valid_articles = 100):
    test_articles = ""
    test_abstracts = ""

    valid_articles = ""
    valid_abstracts = ""

    for category in categories:
        category_articles = open("../" + category + "_NYT/train_test_data/untouched_test_data/" + type_s + "_articles.txt").read().strip()
        # category_articles = category_articles.split("\n")[:num_articles]
        category_articles = category_articles.split("\n")
        
        category_abstracts = open("../" + category + "_NYT/train_test_data/untouched_test_data/" + type_s + "_abstracts.txt").read().strip()
        # category_abstracts = category_abstracts.split("\n")[:num_articles]
        category_abstracts = category_abstracts.split("\n")

        tmp0 = []
        tmp1 = []

        for i in range(0, len(category_articles)):
            if len(tmp0) == num_test_articles:
                break

            if len(category_articles[i].split()) <= threshold:
                tmp0.append(category_articles[i])
                tmp1.append(category_abstracts[i])

        # tmp = shuffle(category_articles, category_abstracts)

        test_articles += "\n" + "\n".join(tmp0)
        test_abstracts += "\n" + "\n".join(tmp1)
        
        if valid == False:
            continue

        tmp0 = []
        tmp1 = []

        for j in range(i, len(category_articles)):
            if len(tmp0) == num_valid_articles:
                break

            if len(category_articles[j].split()) <= threshold:
                tmp0.append(category_articles[j])
                tmp1.append(category_abstracts[j])
            if category_abstracts[j].strip() == "":
                print(category, j)

        valid_articles += "\n" + "\n".join(tmp0)
        valid_abstracts += "\n" + "\n".join(tmp1)
        
    # test_articles = test_articles.replace("\n\n", "\n")
    # test_abstracts = test_abstracts.replace("\n\n", "\n")
    # valid_articles = valid_articles.replace("\n\n", "\n")
    # valid_abstracts = valid_abstracts.replace("\n\n", "\n")
    test_articles = test_articles.strip()
    test_abstracts = test_abstracts.strip()
    valid_articles = valid_articles.strip()
    valid_abstracts = valid_abstracts.strip()

    if not os.path.exists("train_test_data/untouched_test_data"):
        os.mkdir("train_test_data/untouched_test_data")

    if not os.path.exists("train_test_data/untouched_valid_data"):
        os.mkdir("train_test_data/untouched_valid_data")

    f = open("train_test_data/untouched_test_data/" + type_s + "_articles.txt", "w")
    f.write(test_articles.strip())
    f.close()

    f = open("train_test_data/untouched_test_data/" + type_s + "_abstracts.txt", "w")
    f.write(test_abstracts.strip())
    f.close()

    f = open("train_test_data/untouched_valid_data/" + type_s + "_articles.txt", "w")
    f.write(valid_articles.strip())
    f.close()

    f = open("train_test_data/untouched_valid_data/" + type_s + "_abstracts.txt", "w")
    f.write(valid_abstracts.strip())
    f.close()

if __name__ == "__main__":
    # gather_data("by_sentence")
    # gather_data("by_sentence_neg")
    gather_untouched_test_data("by_sentence", valid = True, num_valid_articles=500)
    # gather_untouched_test_data("by_sentence_neg", valid = False)
