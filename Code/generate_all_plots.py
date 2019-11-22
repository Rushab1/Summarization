import matplotlib
import pickle
import os
import tqdm
import numpy as np
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi']= 500 
from matplotlib import pyplot as plt

def double_axis(p, dataset, summarizer, split, name):
    r1 = []
    r2 = []
    x  = []
    for i in p:
        x.append(i[1])
        r1.append(i[2])
        r2.append(i[3])
    
    yticks = np.arange(0, max(r1)+0.05, 0.025)
    xticks = np.arange(0, 1.2, 0.1)
    fig = plt.figure() 
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    
    plt.xlabel("reduction (%)")
    plt.ylabel("Rouge Score")
    plt.plot(x, r1, 'o-')
    plt.plot(x, r2, 'o-')
    plt.grid()
    plt.title(dataset + " : " + split + " : " + summarizer + " : Rouge vs Reduction")
    plt.legend(["Rouge-1", "Rouge-2"])
    plt.savefig(name + ".png")

files = {
            1: "./Results/Source_Analysis/hist_analysis_cnndm_cnndmAllfile_histograms_clf.pkl",
            2: "./Results/Source_Analysis/hist_analysis_cnndm_gigawordAllfile_histograms_clf.pkl",
            3: "./Results/Source_Analysis/hist_analysis_cnndm_nytAllfile_histograms_clf.pkl",
            4: "./Results/Source_Analysis/hist_analysis_cnndm_nytBusinessfile_histograms_clf.pkl",
            5: "./Results/Source_Analysis/hist_analysis_cnndm_nytSciencefile_histograms_clf.pkl",
            6: "./Results/Source_Analysis/hist_analysis_cnndm_nytSportsfile_histograms_clf.pkl",
            7: "./Results/Source_Analysis/hist_analysis_cnndm_nytUSIntlRelationsfile_histograms_clf.pkl",
            8: "./Results/Source_Analysis/hist_analysis_cnndm_ontonotes_mzAllfile_histograms_clf.pkl",
            9: "./Results/Source_Analysis/hist_analysis_cnndm_ontonotes_tcAllfile_histograms_clf.pkl",
            10: "./Results/Source_Analysis/hist_analysis_cnndm_ontonotes_wsjAllfile_histograms_clf.pkl",

            11: "./Results/Source_Analysis/hist_analysis_nyt_cnndmAllfile_histograms_clf.pkl",
            12: "./Results/Source_Analysis/hist_analysis_nyt_gigawordAllfile_histograms_clf.pkl",
            13: "./Results/Source_Analysis/hist_analysis_nyt_nytBusinessfile_histograms_clf.pkl",
            14: "./Results/Source_Analysis/hist_analysis_nyt_nytSciencefile_histograms_clf.pkl",
            15: "./Results/Source_Analysis/hist_analysis_nyt_nytSportsfile_histograms_clf.pkl",
            16: "./Results/Source_Analysis/hist_analysis_nyt_nytUSIntlRelationsfile_histograms_clf.pkl",
            17: "./Results/Source_Analysis/hist_analysis_nyt_ontonotes_mzAllfile_histograms_clf.pkl",
            18: "./Results/Source_Analysis/hist_analysis_nyt_ontonotes_tcAllfile_histograms_clf.pkl",
            19: "./Results/Source_Analysis/hist_analysis_nyt_ontonotes_wsjAllfile_histograms_clf.pkl",
            }

graphs = [ 
            [4, 5, 6, 7     ],  [8, 9, 10], [2, 10, 7, 1], 
            [13, 14 ,15 ,16 ],  [17, 18, 19], [12, 19, 16, 11],
            [1, 11]
            ]

graph_names = ["cnn_nyt", "cnn_ontonotes", "cnn_newswire", "nyt_nyt", "nyt_ontonotes", "nyt_newswire", "cnn_cnn_nyt"]

def source_analysis():
    if not os.path.exists("./Plots/Source_Analysis"):
        os.mkdir("Plots/Source_Analysis")

    cnndm = pickle.load(open("./Results/cnndm_hist_analysis_for_clf.pkl", "rb"))
    nyt = pickle.load(open("./Results/nyt_hist_analysis_for_clf.pkl", "rb"))

    for j in range(0, len(graphs)):
        graph = graphs[j]
        x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        legend = []
        y = []

        for i in graph:
            f = files[i]
            trained_on_dataset = f.split("_")[3]
            dataset = "_".join(f.split("_")[4:]).replace("file_histograms_clf.pkl", "")
            dataset = dataset.replace("All", "")
            dataset = dataset.replace("nyt", "nyt ")
            
            print(f, trained_on_dataset, dataset)

            f = f.split("/")[-1].replace(".pkl", "")

            if trained_on_dataset == "cnndm":
                y.append([])
                for xi in  sorted(list(cnndm[f].keys())):
                    if xi >= 0.3 and xi <= 0.8:
                        y[-1].append(cnndm[f][xi]*100)

            if trained_on_dataset == "nyt":
                y.append([])
                for xi in  sorted(list(nyt[f].keys())):
                    if xi >= 0.3 and xi <= 0.8:
                        y[-1].append(nyt[f][xi] * 100)

            legend.append(trained_on_dataset +":" + dataset)
      
        # fig = plt.figure()
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xticks(x)
        ax.set_yticks(np.arange(0, 109, 10))

        # And a corresponding grid
        ax.grid()

        # Or if you want different settings for the grids:
        ax.grid(which='major', alpha=0.5)

        # plt.axes().set_aspect('equal')
        # plt.axis([0, 100, 0, 100])
        # plt.grid()
        plt.ylabel("Percentage of content-dense sentences")
        plt.xlabel("Classifier Threshold")

        for i in range(0, len(y)):
            ax.plot(x, y[i], '.-')
        plt.legend(legend)

        plt.savefig("./Plots/Source_Analysis/" + graph_names[j] + ".png")


source_analysis()
# RESULTS_DIR = "Results/"
# min_len = 75
# if not os.path.exists("Plots"):
    # os.mkdir("Plots")

# for summarizer in ["fastabs", "opennmt"]:
    # for dataset in ["cnndm", "nyt"]:
        # for split in ["test", "validation"]:

            # if summarizer == "fastabs":
                # summarizer_1 = "CB"
            # elif summarizer == "opennmt":
                # summarizer_1 = "TF"

            # try:
                # p = pickle.load(open("./Results/" + split.upper() + "_" + dataset + "_importance_" + summarizer + str(min_len) + "_rouge.pkl", "rb"))
            # except:
                # continue
            # double_axis(p, dataset, summarizer_1, split, "Plots/" + split.upper() + "_" + dataset + "_importance_" + summarizer + str(min_len) + "_rouge")
