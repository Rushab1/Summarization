import sys
import os
import hashlib
import subprocess
import collections

import json
import tarfile
import io
import pickle as pkl
import nltk
from tqdm import tqdm

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]

# all_train_urls = "url_lists/all_train.txt"
# all_val_urls = "url_lists/all_val.txt"
# all_test_urls = "url_lists/all_test.txt"

nyt_test_dir = "./nyt_data"
nyt_tokenized_orig = "nyt_tokenized_orig"
nyt_tokenized_cleaned_imp = "nyt_tokenized_cleaned_imp"
nyt_tokenized_cleaned_throw = "nyt_tokenized_cleaned_throw"
nyt_url_dir = "./nyt_url_dir"

finished_files_dir = "finished_files"

#split article into sentences at <t> </t>
def extract_sentences(text):
    text = text.replace("</t>", "")
    return text.split("<t>")

def create_nyt_story_file(articles_file, abstracts_file, save_dir, url_save_file, use_nltk = False):
    f = open(articles_file).read().strip()
    f = f.replace("\n\n", "\n<unknown>\n").split("\n")

    a = open(abstracts_file).read().strip()
    a = a.replace("\n\n", "\n<unknown>\n").split("\n")

    print("NYT converting to sumarizer format: file = " + articles_file)

    if not os.path.exists(nyt_url_dir):
        os.mkdir(nyt_url_dir)

    url_file = open(url_save_file, "w")

    for i in tqdm(range(0, len(f))):
        #sent_tokenize the article and abstracts
        try:
            f[i] = f[i].decode("utf-8")
            a[i] = a[i].decode("utf-8")
        except:
            pass

        if use_nltk:
            art_sent = nltk.sent_tokenize(f[i])
            a[i] = a[i].replace(";", ".")
            abs_sent = nltk.sent_tokenize(a[i])
        else:
            art_sent = extract_sentences(f[i])
            try:
                a[i] = a[i].replace(";", ".")
            except Exception as e:
                print("\n\n========================>" + str(len(a)) + ", " + str(len(f)) + "======" + str(i))
                print(e)
                print("========================\n\n")
                sys.exit(0)
            abs_sent = extract_sentences(a[i])

        abs_sent = ["@highlight \n\n" + sent for sent in abs_sent]

        output = "\n\n".join(art_sent)
        output += "\n\n" + "\n\n".join(abs_sent)

        #output = output.encode("utf-8")

        g = open(os.path.join(save_dir, str(i) + ".story"), "w")
        g.write(str(output))
        g.close()

        url_file.write(str(i) + ".story\n")
    url_file.close()

def convert_nyt_to_story(nyt_file, abstracts_file, save_dir, url_file):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    create_nyt_story_file(nyt_file, abstracts_file, save_dir, url_file)


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using
       Stanford CoreNLP Tokenizer
    """
    print("Preparing to tokenize {} to {}...".format(stories_dir,
                                                     tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write(
                "{} \t {}\n".format(
                    os.path.join(stories_dir, s),
                    os.path.join(tokenized_stories_dir, s)
                )
            )
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing {} files in {} and saving in {}...".format(
        len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of
    # files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory {} contains {} files, but it "
            "should contain the same number as {} (which has {} files). Was"
            " there an error during tokenization?".format(
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig)
        )
    print("Successfully finished tokenizing {} to {}.\n".format(
        stories_dir, tokenized_stories_dir))


def read_story_file(text_file):
    with open(text_file, "r") as f:
        # sentences are separated by 2 newlines
        # single newlines might be image captions
        # so will be incomplete sentence
        lines = f.read().split('\n\n')
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def get_art_abs(story_file, abs_file = None):
    """ return as list of sentences"""
    lines = read_story_file(story_file)

    # Lowercase, truncated trailing spaces, and normalize spaces
    lines = [' '.join(line.lower().strip().split()) for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem
    # in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    return article_lines, highlights


def write_to_tar(url_file, out_file, tokenized_stories_dir, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the
       url_file and writes them to a out_file.
    """
    print("Making bin file for URLs listed in {}...".format(url_file))
    url_list = [line.strip() for line in open(url_file)]

    # url_hashes = get_url_hashes(url_list)
    url_hashes = url_list

    story_fnames = [(s+".story").replace(".story.story", ".story") for s in url_hashes]

    num_stories = len(story_fnames)

    # if makevocab:
    vocab_counter = collections.Counter()

    with tarfile.open(out_file, 'w') as writer:
        for idx, s in enumerate(story_fnames):
            if idx % 1000 == 0:
                print("Writing story {} of {}; {:.2f} percent done".format(
                    idx, num_stories, float(idx)*100.0/float(num_stories)))

            # Look in the tokenized story dirs to find the .story file
            # corresponding to this url
            if os.path.isfile(os.path.join(tokenized_stories_dir, s)):
                story_file = os.path.join(tokenized_stories_dir, s)
            else:
                print("Error: Couldn't find tokenized story file {} in either"
                      " tokenized story directory {} . Was there an"
                      " error during tokenization?".format(
                          s, tokenized_stories_dir))

            # Get the strings to write to .bin file
            article_sents, abstract_sents = get_art_abs(story_file)

            # Write to JSON file
            js_example = {}

            js_example['id'] = s.replace('.story', '')

            js_example['article'] = article_sents
            js_example['abstract'] = abstract_sents
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = ' '.join(article_sents).split()
                abs_tokens = ' '.join(abstract_sents).split()
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t != ""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file {}\n".format(out_file))

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
                  'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--articles_file", type=str, required = True)
    args.add_argument("--abstracts_file", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True )
    opts = args.parse_args()

    if not os.path.exists("TMP"):
        os.mkdir("TMP")
    if not os.path.exists("TMP/conversion_output"):
        os.mkdir("TMP/conversion_output")
    if not os.path.exists("TMP/tokenized/"):
        os.mkdir("TMP/tokenized")
    if not os.path.exists(opts.output_dir):
        os.mkdir(opts.output_dir)

    print("Converting NYT data (orig, cleaned (by, imp, throwaway)) into story format")
    convert_nyt_to_story(opts.articles_file, opts.abstracts_file, "TMP/conversion_output/", "TMP/url_file.txt")
    print("Done")


    finished_files_dir = opts.output_dir
    if not os.path.exists(finished_files_dir):
        os.mkdir(finished_files_dir)

    print("Tokenizing stories")
    tokenize_stories("TMP/conversion_output", "TMP/tokenized")
    print("Done")


    print("Read the tokenized stories, do a little " +
            "postprocessing then write to bin files");

    write_to_tar("TMP/url_file.txt", "TMP/tarfile.tar", "TMP/tokenized")

    cmd = ["tar","-C",opts.output_dir,"-xvf", "TMP/tarfile.tar"]
    with open(os.devnull, 'wb') as FNULL:
        subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)

    cmd = "mv " + os.path.join(opts.output_dir,"tarfile", "*") + " " + opts.output_dir
    os.system(cmd)
    cmd = "rm -rf " + os.path.join(opts.output_dir,"tarfile")
    os.system(cmd)

