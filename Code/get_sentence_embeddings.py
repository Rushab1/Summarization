import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow.python.client import device_lib
import argparse
from tqdm import tqdm
import pickle

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def gpu_is_available():
    gpu_list = get_available_gpus()
    return True if len(gpu_list)>0 else False

#Create sentence embedddings using Universal Sentence Decoder
#Creates a dict of <sentence: embedding > for all sentences in the specified file which needs to be a list of sentences
#All train, validation and test files from the domain "All"
#For other domains they can use the same dict for loading embeddings
#Can use a database like pyTables here, but not needed if you have more than 8GB Memory
#sentences_file = location of the Sentences.txt file
def get_sentence_embeddings(sentences_file, save_file):
    print("Reading Sentences text file")
    Sentences = open(sentences_file, "r").read().strip().split("\n")
    print("Done")

    Sentences_dict = {}

    print("Loading USE tf_hub Module")
    embed_module = tf_hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    #Initialize tensorflow
    print("TF initialize")
    if not gpu_is_available():
        session = tf.Session()
    else:
        config = tf.ConfigProto()
        config.graph_options.rewrite_options.shape_optimization = 2
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

    session.run([ tf.global_variables_initializer(), tf.tables_initializer()  ])
    print("New session created")

    n = len(Sentences)
    h = 100000 if gpu_is_available() else 100000

    for i in tqdm(range(0, 1 + int(n/h))):
        s = i*h;  e = min((i+1)*h, n)
        if s>= n: break
        embeddings = session.run(embed_module(Sentences[s:e]))

        for j in range(s, e):
            Sentences_dict[Sentences[j]] = embeddings[j-s]

    f = open(save_file, "wb")
    pickle.dump(Sentences_dict,  f)
    f.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sentences_file", type=str, required=True)
    args.add_argument("--cpu", action = "store_true")
    args.add_argument("--save_file", type=str, default="../TMP/Sentence_embeddings")
    opts = args.parse_args()

    if opts.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    get_sentence_embeddings( opts.sentences_file,  opts.save_file)
