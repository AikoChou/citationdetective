"""

Usage:
	parse_ids.py <pageid-file> [--timeout=<n>]

Options:
    --timeout=<n>    Maximum time in seconds to run for [default: inf].

"""
import os
import re
import sys
import time
import docopt
import numpy as np
import pickle

from utils import *

import mwapi
from mwapi.errors import APIError
import mwparserfromhell

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))


def run_citation_needed(sentences, model, vocab_w2v, section_dict):
    max_len = model.input[0].shape[1].value
    # construct the training data
    X = []
    sections = []

    for text, _, section in sentences:
        # handle abbreviation, special characters
        # transform the text into a word list
        wordlist = text_to_word_list(text)

         # construct the word vector for word list
        X_inst = []
        for word in wordlist:
            if max_len != -1 and len(X_inst) >= max_len:
                break
            X_inst.append(vocab_w2v.get(word, vocab_w2v['UNK']))

        X.append(X_inst)
        sections.append(section_dict.get(section, 0))

    # pad all word vectors to max_len
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')
    sections = np.array(sections)

    return model.predict([X, sections])

def extract(wikitext):
    sentences = []
    paragraphs = {}
    # consider lead section and sections within level 2 headding
    DISCARD_SECTIONS = ["See also", "References", "External links", "Further reading", "Notes"]

    for section in wikitext.get_sections(levels=[2], include_lead=True):
        if not section.filter_headings(): # no heading -> is lead section
            section_name = 'MAIN_SECTION'
            headings = []

        elif section.filter_headings()[0].title.strip() in DISCARD_SECTIONS:
            # ignore sections which content dont need citations
            continue

        else:
            section_name = section.filter_headings()[0].title.strip()
            # store all (sub)heading names
            headings = [h.title for h in section.filter_headings()]

        # split section content into paragraphs
        section_parag = re.split("\n+", section.strip_code())

        for i, paragraph in enumerate(section_parag):
            pid = mkid(section_name+str(i))
            paragraphs[pid] = paragraph
            # clean hyperlinks which strip_code() did not remove
            paragraph = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", paragraph)
            # split paragraph into sentences
            statements = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)

            for s in statements:
                if s in headings: # discard all (sub)heading name
                    continue
                sentences.append((s, pid, section_name.lower()))
    
    return sentences, paragraphs

def query_pageids(pageids):
    session = mwapi.Session('https://en.wikipedia.org/', 'citationdetective', formatversion=2)

    response_docs = session.post(action='query',
                            prop='revisions',
                            rvprop='ids|content',
                            pageids='|'.join(map(str, pageids)),
                            format='json',
                            utf8='',
                            rvslots='*',
                            continuation=True)

    for doc in response_docs:
        for page in doc['query']['pages']:
            revid = page['revisions'][0]['revid']
            title = page['title']
            content = page['revisions'][0]['slots']['main']['content']
            yield (revid, title, content)

def parse(pageids, timeout, model, vocab_w2v, section_dict):
    pageids_list = list(pageids)

    rows = [] # list of [id, sentence, paragraph, section, revid, score]
    results = query_pageids(pageids_list)
        
    for revid, title, wikitext in results:
        start_len = len(rows)
        wikitext = mwparserfromhell.parse(wikitext)
        sentences, paragraphs = extract(wikitext)

        for text, pidx, section in sentences:
            id = mkid(title+section+text)
            row = [id, text, paragraphs[pidx], section, revid]
            rows.append(row)
    
        pred = run_citation_needed(sentences, model, vocab_w2v, section_dict)
        for i, score in enumerate(pred):
            rows[start_len+i].append(score[1])

    print(rows[0])
    return 0

def load_citation_needed():
    # load the vocabulary and the section dictionary
    voc_path = '../citation-needed/embeddings/word_dict_en.pck'
    section_path = '../citation-needed/embeddings/section_dict_en.pck'
    vocab_w2v = pickle.load(open(voc_path, 'rb'))
    section_dict = pickle.load(open(section_path, 'rb'), encoding='latin1')

    # load Citation Needed model
    model = load_model('../citation-needed/models/fa_en_model_rnn_attention_section.h5')
    return model, vocab_w2v, section_dict

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    pageids_file = arguments['<pageid-file>']
    timeout = float(arguments['--timeout'])
    if timeout == float('inf'):
        timeout = None
    start = time.time()
    model, vocab_w2v, section_dict = load_citation_needed()
    with open(pageids_file) as pf:
        pageids = set(map(str.strip, pf))
    ret = parse(pageids, timeout, model, vocab_w2v, section_dict)
    print('all done in %d seconds.' % (time.time()-start))
    sys.exit(ret)
