import os
import re
import sys
import time
import docopt
import numpy as np
import pickle

import config
import database
from utils import *

import mwapi
from mwapi.errors import APIError
import mwparserfromhell

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
K.clear_session()
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))

cfg = config.get_localized_config()
WIKIPEDIA_BASE_URL = 'https://' + cfg.wikipedia_domain

def load_citation_needed():
    # load the vocabulary and the section dictionary
    vocab_w2v = pickle.load(open(cfg.vocb_path, 'rb'))
    section_dict = pickle.load(open(cfg.section_path, 'rb'), encoding='latin1')

    # load Citation Needed model
    model = load_model(cfg.model_path)
    return model, vocab_w2v, section_dict

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

    for section in wikitext.get_sections(levels=[2], include_lead=True):
        if not section.filter_headings(): # no heading -> is lead section
            section_name = 'MAIN_SECTION'
            headings = []

        elif section.filter_headings()[0].title.strip() in cfg.sections_to_skip:
            # ignore sections which content dont need citations
            continue

        else:
            continue # for test, extract function needs to be improved
            section_name = section.filter_headings()[0].title.strip()
            # store all (sub)heading names
            headings = [h.title for h in section.filter_headings()]

        # split section content into paragraphs
        section_parag = re.split("\n+", section.strip_code())

        for i, paragraph in enumerate(section_parag):
            pid = mkid(section_name+str(i))
            paragraphs[pid] = paragraph
            # clean hyperlinks which strip_code() did not remove
            paragraph = re.sub(cfg.hyperlink_regex, " ", paragraph)
            # split paragraph into sentences
            statements = re.split(cfg.sentence_regex, paragraph)

            for s in statements:
                if s in headings: # discard all (sub)heading name
                    continue
                sentences.append((s, pid, section_name.lower()))
    
    return sentences, paragraphs

def query_pageids(pageids):
    session = mwapi.Session(WIKIPEDIA_BASE_URL, cfg.user_agent, formatversion=2)

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

def parse(pageids):
    pageids_list = list(pageids)[:2] # for test, sholud upgrade to multi-threads version
    model, vocab_w2v, section_dict = load_citation_needed()
    
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

    def insert(cursor, r):
        cursor.execute('''
            INSERT INTO statements VALUES(%s, %s, %s, %s, %s, %s)
            ''', r)
    db = database.init_scratch_db()
    for i, r in enumerate(rows):
        print(i, r)
        db.execute_with_retry(insert, r)
    return 0


if __name__ == '__main__':
    start = time.time()
    pageids_file = os.path.expanduser('~/citationdetective/pageids')
    with open(pageids_file) as pf:
        pageids = set(map(str.strip, pf))
    ret = parse(pageids)
    print('all done in %d seconds.' % (time.time()-start))
    sys.exit(ret)