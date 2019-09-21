import numpy as np
import argparse
import pickle
import os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM as OCS
from copy import deepcopy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import multiprocessing as mp

DOMAINS = ["Business", "Sports", "Science", "USIntlRelations", "All"]

def normalize(x):
    x = np.array(x)
    x = x.transpose()

    mu = x.mean(axis = 1)
    sig = x.std(axis=1)
    for i in range(0, len(x)):
        x[i] = (x[i] - mu[i]) / sig[i]

    x = x.transpose()
    return x, mu, sig

def ltrain(x, y, relabeling = True, classifier = "logistic", kernel = "linear"):
    x = deepcopy(x)
    x, mu, sig = normalize(x)
    P = [x[i] for i in range(0, len(x)) if y[i] == 1]
    n = 1.0 * len(P)

    if relabeling == False and classifier == "svm":
        svm = SVC(kernel = kernel)
        svm.fit(x,y)
        return svm, mu, sig

    print("LOGISTIC REGRESSION")
    LR = LogisticRegression(n_jobs = 100, solver='lbfgs')
    LR.fit(x, y)
    if relabeling == False:
        return LR, mu, sig
    print("Done")

    y_pred = LR.predict_proba(P)
    y_pred = y_pred[:, 1]
    e = sum(y_pred/n)

    O = [x[i] for i in range(0, len(x)) if y[i] == -1]
    rand = np.random.permutation(len(O))[0:len(P)]
    O = np.array(O)
    O = O[rand]

    xnew = []
    ynew = []
    weights = []

    y_pred_O= LR.predict_proba(O)
    y_pred_O = y_pred_O[:,1]

    for i in range(0, len(O)):
        w = (y_pred_O[i]/e) / ( (1-y_pred_O[i])/ (1-e) )
        xnew.append(O[i])
        ynew.append(1)
        weights.append(w)

    for i in range(0, len(O)):
        w = weights[i]
        if w <= 1:
            weights.append(1 - w)
            ynew.append(0)
            xnew.append(xnew[i])
        else:
            weights[i] = 1.0

    for p in P:
        xnew.append(p)
        weights.append(1)
        ynew.append(1)

    xnew = np.array(xnew)
    ynew = np.array(ynew)
    weights = np.array(weights)

    rand = np.random.permutation(len(xnew))
    rand = rand[0:7000]

    xnew = xnew[rand]
    ynew = ynew[rand]
    weights = weights[rand]

    print("Learning SVM model")
    svc = SVC(gamma = 'auto', kernel = 'linear', cache_size=10000, probability=True)
    svc.fit(xnew, ynew, weights)
    return svc, mu, sig

def test(X, y, model):
    x = deepcopy(X)
    x = np.array(x)
    mu = model[1]
    sig = model[2]
    m = model[0]

    mu.resize(512, 1)
    sig.resize(512, 1)
    x = x.transpose()
    x =  (x - mu) / sig
    x = x.transpose()

    y_pred = m.predict(x)

    #convert -1 to 0 in y
    if -1 in y:
        y = (np.array(y)+1)/2

    len_pos = len([i for i in y if i == 1])
    len_neg = len([i for i in y if i == 0])
    print("total samples: " + str(len(x))  )
    print("total samples: " + str(len(x))  )
    print("total pos samples expected: " + str(len_pos)  )
    print("total neg samples expected: " + str(len_neg)  )

    #convert any -1 in y_pred to 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == -1:
            y_pred[i] = 0

    print("POSITIVE CLASS")
    pr = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print("precision", pr)
    print("recall", rec)
    print("f1", f1)
    print("accuracy", acc)

    print("NEGATIVE CLASS")
    yneg = - np.array(deepcopy(y)) + 1
    y_pred_neg = - np.array(deepcopy(y_pred)) + 1

    pr = precision_score(yneg, y_pred_neg)
    rec = recall_score(yneg, y_pred_neg)
    acc = accuracy_score(yneg, y_pred_neg)
    f1 = f1_score(yneg, y_pred_neg)

    print("precision", pr)
    print("recall", rec)
    print("f1", f1)
    print("accuracy", acc)
    return y_pred

def main(dataset, domain, type_s="importance", relabeling=True, classifier="logistic"): #classifier arg Used only if relabeling is False
    print("\n______________________________\n{} - training {}".format(dataset, domain))

    #loading Data
    load_file = os.path.join("../Data/Processed_Data/", dataset, domain, type_s, "train_data.pkl")

    print(load_file)
    dct = pickle.load(open(load_file, "rb"))

    x = dct['embedding']
    y = dct['label']

    rand = np.random.permutation(len(x))
    x = np.array(x)[rand]
    y = np.array(y)[rand]


    #Training Model
    if type_s == "importance":
        model = ltrain(x, y, relabeling = relabeling, classifier = classifier)
    else:
        print("Setting relabeling to False ")
        model = ltrain(x, y, relabeling = False, classifier = classifier)

    save_file = os.path.join("../Data/Processed_Data/", dataset, domain, type_s, "model.pkl")

    pickle.dump(model, open(save_file, "wb"))

    print("Done\n______________________________\n{} - testing {}".format(dataset, domain))

    #Testing model
    a = pickle.load(open(os.path.join("../Data/Processed_Data/", dataset, domain, type_s, "test_data.pkl"), "rb"))
    X_test = a['embedding']
    Y_test = a['label']

    rand = np.random.permutation(len(X_test))
    X_test = np.array(X_test)[rand]
    Y_test = np.array(Y_test)[rand]

    test(X_test, Y_test, model)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dataset", type=str, default="nyt")
    args.add_argument("-type_s", type=str, default="importance")
    args.add_argument("-relabeling", type=int, default=1)
    opts = args.parse_args()

    dataset = opts.dataset
    type_s = opts.type_s
    relabeling = opts.relabeling

    if dataset in ["cnn", "gigaword", "cnndm"]:
        DOMAINS = ["All"]

    #Running on domains in parallel - Saves time
    pool = mp.Pool()
    jobs = []

    for domain in DOMAINS:
        job = pool.apply_async(main, (dataset, domain, type_s, relabeling, "logistic" ))
        jobs.append(job)

    for job in jobs:
        job.get()

    pool.close()
    pool.join()
