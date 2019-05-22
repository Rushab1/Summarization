import pickle
import argparse
import os

def summarize_lead(
        articles, 
        lengths):

    summaries = []
    n = len(articles)

    for i in range(0, n):
        summaries.append(
                " ".join(articles[i].strip().split()[:lengths[i]])
                )

    return summaries

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-fix_length", action='store_true')
    args.add_argument("-length", type=int, default=35)
    args.add_argument("-articles_file", type=str)
    args.add_argument("-cleaned_articles_file", type=str)
    args.add_argument("-neural_summaries_file", type=str)
    args.add_argument("-original_summaries_file", type=str)
    # args.add_argument("-original_summaries_file", type=str)
    opts = args.parse_args()

    articles = open(opts.articles_file).read().strip().split("\n")
    cleaned_articles = open(opts.cleaned_articles_file).read().strip().split("\n")
    n = len(articles)

    assert(len(cleaned_articles) <= len(articles))

    if opts.fix_length:
        lengths =  np.ones(n) * opts.lengths
    else:
        neural_abstracts = open(opts.neural_summaries_file).read().strip()
        neural_abstracts = neural_abstracts.replace("<t>", "")
        neural_abstracts = neural_abstracts.replace("</t>", "")
        neural_abstracts = neural_abstracts.split("\n")
        
        assert(len(neural_abstracts) <= len(articles))
        assert(len(cleaned_articles) <= len(articles))
        assert(len(neural_abstracts) <= len(cleaned_articles))
        lengths = []

        for i in range(0, len(neural_abstracts)):
            lengths.append(len(neural_abstracts[i].split()))

        articles = articles[:len(neural_abstracts)]
        cleaned_articles = cleaned_articles[:len(neural_abstracts)]
        
    print(len(neural_abstracts), len(cleaned_articles), len(articles))
    lead_summaries = summarize_lead(articles, lengths)
    cleaned_lead_summaries = summarize_lead(cleaned_articles, lengths)

    orig_abstracts = open(opts.original_summaries_file).read().strip()
    orig_abstracts = orig_abstracts.replace("<t>", "")
    orig_abstracts = orig_abstracts.replace("</t>", "")
    orig_abstracts = orig_abstracts.split("\n")

    try:
        os.mkdir("TMP")
    except Exception as e:
        print(e)
        pass

    open("TMP/clean_lead.txt", "w").write("\n".join(cleaned_lead_summaries))
    open("TMP/orig.txt", "w").write("\n".join(orig_abstracts))
    open("TMP/unclean_lead.txt", "w").write("\n".join(lead_summaries))

    print("UNCLEAN Lead summaries of length equal to CLEANED neural summaries ")
    os.system("files2rouge ./TMP/unclean_lead.txt ./TMP/orig.txt")

    print("CLEANED Lead summaries of length equal to CLEANED neural summaries ")
    os.system("files2rouge ./TMP/clean_lead.txt ./TMP/orig.txt")
