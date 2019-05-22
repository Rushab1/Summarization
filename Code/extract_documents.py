import os
import sys
import pickle
import re
import nltk

dir_orig = '/project/cis/nlp/data/corpora/nytimes/data/'
dir_proc = '/nlp/users/yinfeiy/informative_summary/data/full/abstract/'
categories = ['Business', 'Science', 'Sports', 'USIntlRelations']

def clean_and_process_file(fname, year):
    f = fname.split("_")
    # /project/cis/nlp/data/corpora/nytimes/data/1999/01/07/1075412.xml
    f = open(dir_orig + year + "/" + \
           f[1] + "/" + f[2] +"/" + \
           f[3].split('.')[0] + ".xml")
    f = f.read()

    abstract = f.split('<abstract>')
    if len(abstract) == 1:
        # continue
        return ""

    abstract = abstract[1].split('</abstract>')
    f = abstract[1]
    abstract = abstract[0]
    # abstract = re.sub(";.*?$", "", abstract)
    abstract = abstract.replace("<p>", "")
    abstract = abstract.replace("</p>", "")

    #Clean abstract
    abstract = abstract.strip()
    tmp = abstract.split(';')

    tmp1 = []
    while len(tmp) > 1 and len(tmp[-1]) <= 20:
        tmp1.append(tmp[-1])
        del tmp[-1]
        if len(tmp) == 0:
            print("__________________________")
            print(f_orig_name)
            print(tmp1)
            print("__________________________")
            


    lead_paragraph = f.split("<block class=\"lead_paragraph\">")
    
    lead_paragraph = lead_paragraph[0].split("</block>")[0]
    lead_paragraph = lead_paragraph.replace("<p>", "")
    lead_paragraph = lead_paragraph.replace("</p>", "")

    full_text = f.split("<block class=\"full_text\">")
    full_text = full_text[1]
    full_text = full_text.replace("<p>", "")
    full_text = full_text.replace("</p>", "")
    pattern = re.compile("\<\/block\>.*", re.DOTALL)
    full_text = re.sub( pattern, "", full_text)
    full_text = re.sub('\'\'', r'"', full_text)
    full_text = re.sub('([.,!?()\'"])', r' \1 ', full_text)
    full_text = full_text.strip()
    full_text = re.sub("\t", " ", full_text)
    full_text = re.sub(" [ ]*", " ", full_text)

    full_text = full_text.decode("utf-8")
    full_text = full_text.encode("ascii", "ignore")
    # full_text = full_text.split("\n")
    full_text = full_text.replace("\n", " ")
    full_text = nltk.sent_tokenize(full_text)


    # for i in range(0, len(full_text)):
        # full_text[i] = full_text[i].strip()
    # full_text = f.split("</block>")

    return_dct = {
            "abstract": abstract,
            "lead_paragraph": lead_paragraph,
            "full_text": full_text
            }
    return return_dct


def extract_documents_year(cat, year):
    print(len(os.listdir(dir_proc + cat + "/" + year)))
    
    cnt = 0
    for f in os.listdir(dir_proc + cat + "/" + year):
        f_orig_name = f
        cnt += 1
        sys.stdout.write("\r" + str(cnt))
        sys.stdout.flush()
        f = clean_and_process_file(f,year)

def create_file_list():
    fnames = {}
    for cat in categories:
        fnames[cat] = []

    cnt = 0
    for cat in categories:
        for year in os.listdir(dir_proc + cat):
            print(cat + " : " + year)
            extract_documents_year(cat, year)

if __name__ == "__main__":
     create_file_list()

