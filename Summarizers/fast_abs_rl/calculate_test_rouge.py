from files2rouge import files2rouge
import os
from copy import deepcopy

files_dict = {}

def get_type_from_file(fname):
    fname = fname.replace(".txt", "")
    fname = fname.split("_")
    category = fname[0]
    min_length = fname[1]
    threshold = fname[2]
    type_s = "_".join(fname[3:-1])
    file_type = fname[-1]

    return (category, min_length, threshold, type_s), file_type

directory = "./Data/nyt_output/"
for f in os.listdir(directory):
    fread = open(directory + f).read()
    g = open(directory + f, "w")
    fread = fread.strip().replace("<t>", "")
    fread = fread.replace("</t>", "")
    g.write(fread)
    g.close()

    f_details, file_type = get_type_from_file(deepcopy(f))
    if f_details not in files_dict:
        files_dict[f_details] = {}

    print(file_type)
    files_dict[f_details][file_type + '_file'] = f

g = open("Data/nyt_rouge_results.txt", "w")
for file in files_dict:
    print(file)
    orig = directory + files_dict[file]["orig_file"]
    cleaned = directory + files_dict[file]["cleaned_file"]
    abstracts = directory + files_dict[file]["abstracts_file"]

    print(orig, cleaned, abstracts)
    f = files2rouge.run(orig, abstracts)
    output = "\t".join(file)

    output += "\t" + str(f["rouge-1"]["average_f"])
    output += "\t" + str(f["rouge-2"]["average_f"])
    output += "\t" + str(f["rouge-l"]["average_f"])

    f = files2rouge.run(cleaned, abstracts)

    output += "\t" + str(f["rouge-1"]["average_f"])
    output += "\t" + str(f["rouge-2"]["average_f"])
    output += "\t" + str(f["rouge-l"]["average_f"])

    output += "\n\n"

    g.write(output)

g.close()
