import os
import re
import sys
import time
import docopt
import numpy as np
import pickle
import nltk

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
    # load the vocabulary and the section title dictionary
    vocab_w2v = pickle.load(open(cfg.vocb_path, 'rb'))
    section_dict = pickle.load(open(cfg.section_path, 'rb'), encoding='latin1')

    # load Citation Needed model
    model = load_model(cfg.model_path)
    return model, vocab_w2v, section_dict

def run_citation_needed(sentences, model, vocab_w2v, section_dict):
    max_len = cfg.word_vector_length
    
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
        headings = section.filter_headings()
        # Lead section
        if not headings: 
            section_name = 'MAIN_SECTION'
            headings = []
        else:
            section_name = headings[0].title.strip()
            if section_name in cfg.sections_to_skip:
                continue

        # Remove heading
        for heading in headings:
            section.remove(heading)

        # Remove [[File:...]] and [[Image:...]] cases in wikilinks
        # https://github.com/earwig/mwparserfromhell/issues/136
        # Modified from:
        # https://github.com/earwig/earwigbot/blob/develop/earwigbot/wiki/copyvios/parsers.py#L140
        prefixes = ("file:", "image:", "category:")
        for link in section.filter_wikilinks():
            if link.title.strip().lower().startswith(prefixes):
                section.remove(link)

        # Remove table 
        # https://github.com/earwig/mwparserfromhell/issues/93
        for table in section.filter_tags(matches=lambda node: node.tag == "table"):
            section.remove(table)

        # Remove reference
        for ref_tag in section.filter_tags(matches=lambda node: node.tag == "ref"):
            section.remove(ref_tag)

        section = section.strip_code()
        section_parag = re.split("\n+", section)

        for i, paragraph in enumerate(section_parag):
            pid = mkid(section_name+str(i))
            paragraphs[pid] = paragraph
            for sent in nltk.tokenize.sent_tokenize(paragraph):
                sentences.append((sent, pid, section_name.lower()))

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