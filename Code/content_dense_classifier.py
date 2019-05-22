import argparse
import os
import pickle
import tensorflow_hub as tf_hub
import tensorflow as tf
import numpy as np
from sklearn.svm import OneClassSVM as ocs
import re

# data_dir = "../pair_sent_selected/Business/"
score_threshold_up = 14.0
score_threshold_down = 10.0
create_embeddings = False
len_test_data = 1000

def get_data(news_type, data_dir, create_embeddings, create_test_data):
    global score_threshold_up, score_threshold_down, len_test_data

    if not os.path.exists("modelfiles/" + news_type):
        os.mkdir("modelfiles/" + news_type)

    training_data_str = [[], []]
    training_data_embed = [[], []]
    sent_list_embed = []

    meta = open(data_dir+"meta.txt").read().strip().split("\n")
    sent_list=open(data_dir+"sent.txt").read().strip().split("\n")
    n = len(meta)
    

    #Creating Training data 
    pos = []
    neg = []

    for i in range(0, n):
        line = meta[i]
        line = line.split(" || ")
        code = line[0] + "_" + line[2]
        score = float(line[3])

        if score >= score_threshold_up:
            training_data_str[0].append(sent_list[i])
            training_data_str[1].append(1)
            pos.append(i)

        elif score <= score_threshold_down:
            training_data_str[0].append(sent_list[i])
            training_data_str[1].append(-1)
            neg.append(i)

        else:
            #Don't use this data for training
            pass
    
    n = len(training_data_str[0])
    h = 2000
    max_embeddings_per_file = 2000

    test_indices = np.random.permutation(len(meta))
    test_indices = test_indices[0:len_test_data]
    test_embed =[]
    test_str = []

    with tf.Session() as session:
        embed = tf_hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        session.run([ tf.global_variables_initializer(), \
                          tf.tables_initializer()  ])
       
        #Create Embeddings
        if create_embeddings:
            cnt = 0
            num_files = 0

            for i in range(0, int(1+ 1.0*n/h)):
                s = i*h
                e = min(n, (i+1)*h)
                print(s,e)
                if s >= n:
                    continue

                tmp = session.run(embed(training_data_str[0][s:e]))
                training_data_embed[0].extend(tmp)
                
                cnt += len(tmp)
                
                training_data_embed[1].extend( \
                        training_data_str[1][s:e])

                if cnt >= max_embeddings_per_file or \
                        e == n:
                    dumpfile = open("modelfiles/" + news_type + \
                                    "/training_" + str(num_files) \
                                    + ".pkl", "wb")
                    pickle.dump(training_data_embed, dumpfile)

                    del training_data_embed
                    training_data_embed = [[], []]
                    cnt = 0
                    num_files += 1
    
        create_single_file(news_type)

        #Create test data
        if create_test_data:
            for i in range(0, len_test_data):
                test_str.append(sent_list[test_indices[i]])

            for i in range(0, int(1.0+1.0*len_test_data/h)):
                s = i*h
                e = min(len_test_data, (i+1)*h)

                if s >= len_test_data:
                    continue

                print(s,e)
                tmp = session.run(embed(test_str[s:e]))
                test_embed.extend(tmp)

            dct = {"test_sentences": test_str,
                    "test_embeddings": test_embed}

            pickle.dump(dct, open("modelfiles/" + news_type +"/"+ \
                "test_data.pkl", "wb"))
    



def create_single_file(news_type):
        #Create Single File
        x = []
        y = []

        print("Loading Files")
        for filename in os.listdir("modelfiles/" + news_type):
            if not re.match("training.*", filename):
                continue
            filename = "modelfiles/" + news_type + "/" + filename
            a = pickle.load(open(filename, "rb"))
            x.extend(a[0])
            y.extend(a[1])
            # os.rmdir("modelfiles/" + news_type + "/" + filename)

        dct = {"x": x, "y": y}
        pickle.dump(dct, open("modelfiles/" + news_type +\
                "_training_data.pkl", "w"))


        print "Done"

def train(news_type):
    # x = []
    # y = []

    # print("Loading Files")
    # for filename in os.listdir("modelfiles/" + news_type):
        # if not re.match("training.*", filename):
            # continue
        # filename = "modelfiles/" + news_type + "/" + filename
        # a = pickle.load(open(filename, "rb"))
        # x.extend(a[0])
        # y.extend(a[1])
    
    # dct = {"x": x, "y": y}
    # pickle.dump(dct, open("modelfiles/" + news_type +\
                # "_training_data.pkl", "wb"))

    dct = pickle.dump(open("modelfiles/" + news_type +\
                "_training_data.pkl", "rb"))
    print("Learning Model")
    model = ocs(kernel="linear")
    pos = [x[i] for i in range(0, len(x)) if y[i] == 1]
    model.fit(pos)
    return model

def predict(news_type):
    model = pickle.load(open("./modelfiles/" + news_type + "_model.pkl", "rb"))
    x = pickle.load(open("./modelfiles/" + news_type + "/test_data.pkl", "rb"))
    x = x['test_embeddings']
    return model.predict(x)

def main(news_type = 'Business',
         create_embeddings = False,
         create_test_data = True):

    data_dir = '../pair_sent_selected/' + news_type + '/'

    if create_embeddings or create_test_data:
        get_data(news_type, data_dir, \
            create_embeddings, create_test_data)

    # if create_embeddings:
        # dct = {"training_data_sentences": a[1],
                # "training_data_embeddings": a[0]
                # }

        # pickle.dump(dct, open("modelfiles/" + \
                # news_type + "_training_data.pkl", "wb"))

    # if create_test_data:
        # dct = {"test_data_sentences": a[2],
                # "test_data_embeddings": a[3]
                # }


    model = train(news_type)
    pickle.dump(model, open("modelfiles/" + \
            news_type + "_model.pkl", "wb"))
    predict(news_type)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-news_type", 
            type = str, default = "Sports")
    args.add_argument("-create_embeddings", type = int, 
            default = 0)
    args.add_argument("-create_test_data", type = int,
            default = 1)
    opts = args.parse_args()

    create_single_file(opts.news_type)
    main(opts.news_type,
         opts.create_embeddings, 
         opts.create_test_data)










