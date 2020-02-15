import os
import re
import sys
import time
import docopt
import numpy as np
import pickle
import multiprocessing
import functools

import cddb
import config
from utils import *

import mwapi
import mwparserfromhell
import nltk

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
K.clear_session()
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))

cfg = config.get_localized_config()
WIKIPEDIA_BASE_URL = 'https://' + cfg.wikipedia_domain

# load the vocabulary and the section title dictionary
vocab_w2v = pickle.load(open(cfg.vocb_path, 'rb'))
section_dict = pickle.load(open(cfg.section_path, 'rb'), encoding='latin1')

# load Citation Needed model
model = load_model(cfg.model_path)

def run_citation_needed(sentences):
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

def clean_wikicode(section):
    # Remove [[File:...]] and [[Image:...]] in wikilinks
    # https://github.com/earwig/mwparserfromhell/issues/136
    # Modified from:
    # https://github.com/earwig/earwigbot/blob/develop/earwigbot/wiki/copyvios/parsers.py#L140
    prefixes = ("file:", "image:", "category:")
    for link in section.filter_wikilinks():
        if link.title.strip().lower().startswith(prefixes):
            try:
                section.remove(link)
            except ValueError:
                continue

    # Remove tables
    # https://github.com/earwig/mwparserfromhell/issues/93
    for tbl in section.filter_tags(matches=lambda node: node.tag == "table"):
        try:
            section.remove(tbl)
        except ValueError:
            continue

    # Only remove infobox template, because we don't
    # want to remove any template in a sentence
    for tpl in section.filter_templates():
        if tpl.name.strip().lower().startswith("infobox"):
            try:
                section.remove(tpl)
            except ValueError:
                continue

def extract(wikitext):
    sentences = []
    paragraphs = {}

    for section in wikitext.get_sections(levels=[2], include_lead=True):
        headings = section.filter_headings()
        if not headings: 
            section_name = 'MAIN_SECTION'
        else:
            section_name = headings[0].title.strip()
            if section_name in cfg.sections_to_skip:
                continue
            for heading in headings:
                section.remove(heading)

        clean_wikicode(section)

        for i, paragraph in enumerate(re.split("\n+", str(section))):
            pid = mkid(section_name+str(i))
            paragraphs[pid] = paragraph
            for sent in nltk.tokenize.sent_tokenize(paragraph):
                if sent.strip()[0] in "|!<{":
                    # Not a sentence if it starts with
                    # special characters
                    continue
                if len(nltk.tokenize.word_tokenize(sent)) < cfg.sentence_min_word_count:
                    # Not a sentence if it is too short
                    continue
                sentences.append((sent, pid, section_name.lower()))
    return sentences, paragraphs

def query_pageids(pageids):
    session = mwapi.Session(WIKIPEDIA_BASE_URL, cfg.user_agent, formatversion=2)

    response = session.post(action='query',
                            prop='revisions',
                            rvprop='ids|content',
                            pageids='|'.join(map(str, pageids)),
                            format='json',
                            utf8='',
                            rvslots='*',
                            continuation=True)

    for doc in response:
        for page in doc['query']['pages']:
            if 'revisions' not in page:
                continue
            revid = page['revisions'][0]['revid']
            if 'title' not in page:
                continue
            title = page['title']
            content = page['revisions'][0]['slots']['main']['content']
            yield (revid, title, content)

def with_max_exceptions(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwds):
        try:
            return fn(*args, **kwds)
        except:
            traceback.print_exc()
            self.exception_count += 1
            if self.exception_count > MAX_EXCEPTIONS_PER_SUBPROCESS:
                print('Too many exceptions, quitting!')
                raise
    return wrapper

@with_max_exceptions
def work(pageids):
    rows = []
    results = query_pageids(pageids)
    for revid, title, wikitext in results:
        start_len = len(rows)
        wikicode = mwparserfromhell.parse(wikitext)
        sentences, paragraphs = extract(wikicode)

        for text, pidx, section in sentences:
            row = [text, paragraphs[pidx], section, revid]
            rows.append(row)

        pred = run_citation_needed(sentences)
        for i, score in enumerate(pred):
            rows[start_len+i].append(score[1])

    def insert(cursor, r):
        cursor.execute('''
            INSERT INTO statements (statement, context, section, rev_id, score)
            VALUES(%s, %s, %s, %s, %s)
            ''', r)
    db = cddb.init_scratch_db()
    for r in rows:
        try:
            db.execute_with_retry(insert, r)
        except:
            # This may be caused by the size
            # of a sentence or paragraph that
            # exceed the maximum limit
            continue

def parse(pageids, timeout):
    pool = multiprocessing.Pool(processes=2)
    tasks = []
    batch_size = 32
    pageids_list = list(pageids)
    for i in range(0, len(pageids_list), batch_size):
        tasks.append(pageids_list[i:i+batch_size])

    result = pool.map_async(work, tasks)
    pool.close()

    if timeout is not None:
        result.wait(timeout)
    else:
        result.wait()
    if not result.ready():
        print('timeout, canceling the process pool!')
        pool.terminate()

    pool.join()
    try:
        result.get()
        ret = 0
    except Exception as e:
        print('Too many exceptions, failed!')
        ret = 1
    return ret

if __name__ == '__main__':
    start = time.time()
    pageids_file = os.path.expanduser('~/citationdetective/pageids')
    with open(pageids_file) as pf:
        pageids = set(map(str.strip, pf))
    ret = parse(pageids, None)
    print('all done in %d seconds.' % (time.time()-start))
    sys.exit(ret)
