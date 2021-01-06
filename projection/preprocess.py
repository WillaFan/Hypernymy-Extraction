#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import codecs
import collections
from tqdm import tqdm

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='Log.log', level=logging.DEBUG, format=LOG_FORMAT)


SUBTASK="2B"

"""
Preprocess the corpus
"""
def is_ngram(vocab):
    trigrams = set()
    bigrams = set()
    unigrams = set()
    for term in vocab:
        nb_words = len(term.split())
        if nb_words == 3:
            trigrams.add(term)
        elif nb_words == 2:
            bigrams.add(term)
        elif nb_words == 1:
            unigrams.add(term)
        else:
            msg = "Error: '{}' is not unigram, bigram or trigram".format(term)
            raise ValueError(msg)
    return unigrams, bigrams, trigrams

def n_gram_count(corpus_path, vocab):
    logging.info("Counting lines in corpus...")
    print("Counting lines in corpus...")
    nb_lines = sum(1 for line in open(corpus_path, encoding="utf-8"))
    logging.info("Counting n-gram frequencies in corpus...")
    print("Counting n-gram frequencies in corpus...")
    term_to_freq_in = collections.defaultdict(int)
    line_count = 0
    with open(corpus_path, encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            line_count += 1
            # output every 100000 lines
            if line_count % 100000 == 0:
                msg = "{}/{} lines processed.".format(line_count, nb_lines)
                msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_in), len(vocab))
                logging.info(msg)
            line = line.strip().replace("_", "")
            words = [w.lower() for w in line.split()]
            for n in [1,2,3]:
                for i in range(len(words)+n-1):
                    term = " ".join(words[i:i+n])
                    if term in vocab:
                        term_to_freq_in[term] += 1
    msg = "{}/{} lines processed.".format(line_count, nb_lines)
    msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_in), len(vocab))
    print(msg)
    return term_to_freq_in

def get_formatted_sample(strings, max_sampled):
    sub = strings[:max_sampled]
    if len(strings) > max_sampled:
        sub.append("... ({}) more".format(len(strings)-max_sampled))
    return ", ".join(sub)

def extract_ngrams(tokens, n, ngram_vocab, term_to_freq):
    """ Given a list of tokens and a vocab of n-grams, extract list of
    non-overlapping n-grams found in tokens, using term frequency to
    resolve overlap, in a way that favours low-frequency n-grams.
    Args:
    - Tokens: list of tokens
    - n: size of n-grams
    - ngram_vocab: set of target n-grams
    - term_to_freq: dict that maps terms to their frequency
    Returns: 
    - List of (index, term) tuples, where index is the index of the
      first token of each n-gram, and term is the n-gram (joined with
      spaces).
    """
    ngrams_found = []
    for i in range(len(tokens)-n+1):
        term = " ".join(tokens[i:i+n])
        if term in ngram_vocab:
            ngrams_found.append((i,term))
    if len(ngrams_found) < 2:
        return ngrams_found
    # Eliminate overlap
    ngrams_filtered = ngrams_found[:1]
    for (start, term) in ngrams_found[1:]:
        prev_start, prev_term = ngrams_filtered[-1]
        if start - prev_start < n:
            if term not in term_to_freq or term_to_freq[term] < term_to_freq[prev_term]:
                ngrams_filtered[-1] = (start, term)
        else:
            ngrams_filtered.append((start, term))
    return ngrams_filtered

def get_indices_unmasked_spans(mask):
    """ Given a mask array (list where masked elements are evaluated
    as True and unmasked elements are evaluated as False), return
    spans of unmasked list items."""
    spans = []
    start = 0
    while start < len(mask):
        if mask[start]:
            start += 1
            continue
        end = start
        for i in range(start+1, len(mask)):
            if mask[i]:
                break
            else:
                end = i
        spans.append((start, end))
        start = end + 1
    return spans

def preprocess_corpus(input_file, output_file, queries, candidates, term_to_freq_in, replace_OOV=True):
    logging.info("Processing corpus...")
    print("Processing corpus...")
    term_to_freq_out = collections.defaultdict(int)
    line_count = 0
    vocab = candidates.union(queries)
    unigrams, bigrams, trigrams = is_ngram(vocab)
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w",encoding="utf-8") as f_out:
        for line in tqdm(f_in):
            line_count += 1
            if line_count % 100000 == 0:
                msg = "{} lines processed.".format(line_count)
                msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_out), len(vocab))
                logging.info(msg)
            line = line.strip().replace("_", "")
            words = [w.lower() for w in line.split()]
            # Make array indicating the length of the term found at each position
            term_lengths = [0 for _ in range(len(words))]
            # Make array indicating which indices are masked because a
            # term has already been found there
            masked_indices = [0 for _ in range(len(words))]
            
            # Check for trigrams
            trigrams_found = extract_ngrams(words, 3, trigrams, term_to_freq_in)
            for (i, term) in trigrams_found:
                term_lengths[i] = 3
                term_to_freq_out[term] += 1
                masked_indices[i] = 1
                masked_indices[i+1] = 1
                masked_indices[i+2] = 1
            # Check for bigrams
            for (beg, end) in get_indices_unmasked_spans(masked_indices):    
                bigrams_found = extract_ngrams(words[beg:end+1], 2, bigrams, term_to_freq_in)                
                for (i, term) in bigrams_found:
                    term_lengths[beg+i] = 2
                    term_to_freq_out[term] += 1
                    masked_indices[beg+i] = 1
                    masked_indices[beg+i+1] = 1
            # Check for unigrams
            for (beg, end) in get_indices_unmasked_spans(masked_indices):    
                for i in range(beg,end+1):
                    term = words[i]
                    if term in unigrams:
                        term_to_freq_out[term] += 1
                        term_lengths[i] = 1
            # Write sentence
            norm_terms = []
            i = 0
            while i < len(term_lengths):
                n = term_lengths[i] 
                if n > 1:
                    norm_terms.append("_".join(words[i:i+n]))
                    i += n
                else:
                    if replace_OOV and n == 0:
                        norm_term = "<UNK>"
                    else:
                        norm_term = words[i]
                    norm_terms.append(norm_term)
                    i += 1
            sent = " ".join(norm_terms)
            f_out.write(sent+"\n")
    msg = "{} lines processed.".format(line_count)
    msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_out), len(vocab))
    logging.info(msg)
    missing_q = [w for w in queries if term_to_freq_out[w] == 0]
    missing_c = [w for w in candidates if term_to_freq_out[w] == 0]
    print("# of missing queries in output: {}".format(len(missing_q)))
    max_shown = 200
    if len(missing_q):
        msg = "Examples: {}".format(get_formatted_sample(sorted(missing_q), max_shown))
        print(msg)
    print("# of missing candidates in output: {}".format(len(missing_c)))
    if len(missing_c):
        msg = "Examples: {}".format(get_formatted_sample(sorted(missing_c), max_shown))
        print(msg)

    msg = "\nWrote corpus to '{}'".format("./f_out_preprocessed_corpus.txt")
    print("Finished...")
    print(msg)
    
    return term_to_freq_out

def write_freq(vocab_out_name, term_to_freq_out):
    path_output = "./" + vocab_out_name + ".vocab"+".txt"
    with open(path_output, "w", encoding="utf-8") as f:
        for term, freq in sorted(term_to_freq_out.items(), key=lambda x:x[0]):
            # Normalize term
            term_norm = "_".join(term.split())
            f.write("{}\t{}\n".format(term_norm, freq))
    msg = "Wrote vocab to '{}'".format(path_output)
    print(msg)

    
"""
Preprocess data
"""

    

class Dataset(object):
    def __init__(self,subtask=SUBTASK):
        self.subtask=subtask
    
    def expand_subtask_name(self):
        """ Given short name of subtask, return long name. """
        if self.subtask == "1A":
            return "1A.english"
        elif self.subtask == "1B":
            return "1B.italian"
        elif self.subtask == "1C":
            return "1C.spanish"
        elif self.subtask == "2A":
            return "2A.medical"
        elif self.subtask == "2B":
            return "2B.music"
        else:
            msg = "Unrecognized subtask name '{}'".format(self.subtask)
            raise ValueError(msg)

    def load_candidates(self, path, normalize=True):
        """Given the path of a list of candidate hypernyms, return list of
        candidates.
        """
        with codecs.open(path, "r", encoding="utf-8") as f:
            candidates = []
            for line in f:
                c = line.strip()
                if len(c):
                    if normalize:
                        c.lower().replace(" ", "_")
                    candidates.append(c)
            return candidates

    def load_queries(self, path, normalize=True):
        """Given the path of a query file, return list of queries and list
        of query types.
        """
        with codecs.open(path, "r", encoding="utf-8") as f:
            queries = []
            query_types = []
            for line in f:
                q, qtype = line.strip().split("\t")
                if normalize:
                    q = q.lower().replace(" ", "_")
                queries.append(q)
                query_types.append(qtype)
        return queries, query_types
    
    def load_hypernyms(self, path, normalize=True):
        """Given the path of a hypernyms file, return list of lists of
        hypernyms.
        """
        with codecs.open(path, "r", encoding="utf-8") as f:
            hypernyms = []
            for line in f:
                h_list = line.strip().split("\t")
                if normalize:
                    h_list = [h.lower().replace(" ", "_") for h in h_list]
                hypernyms.append(h_list)
        return hypernyms

    def load_vocab(self, path_data, lower_queries=False):
        """ Given the path of the directory containing the data for
        SemEval 2018 task 9 and the name of a subtask (e.g. 1A), load
        candidates and queries (training, trial, and test). Return set of
        candidates and set of queries.
        """
        dataname = self.expand_subtask_name()
        path = "{}/vocabulary/{}.vocabulary.txt".format(path_data, dataname)
        logging.info("Loading candidates from vocabulary ...")
        print("Loading candidates from vocabulary ...")
        candidates = set(self.load_candidates(path, normalize=False))
        # path = "{}/vocabulary/{}.vocabulary.txt".format(path_data, dataname)
        queries = set()
        for part in ["training", "trial", "test"]:
            logging.info("Loading queries from '"+path+"' ...")
            print("Loading queries from '"+path+"' ...")
            path = "{}/{}/data/{}.{}.data.txt".format(path_data, part, dataname, part)
            q, _ = self.load_queries(path, normalize=False)
            if lower_queries:
                q = [query.lower() for query in q]
            queries.update(q)
        return candidates, queries

