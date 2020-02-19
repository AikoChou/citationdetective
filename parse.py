import os
import re
import sys
import time
import docopt
import numpy as np
import pickle
from collections import defaultdict

import cddb
import config
from utils import *

import mwapi
import mwparserfromhell
import nltk.data
from nltk.tokenize import word_tokenize

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
K.clear_session()
K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))

cfg = config.get_localized_config()
WIKIPEDIA_BASE_URL = 'https://' + cfg.wikipedia_domain

# Tweak the NLTK's pre-trained English sentence tokenizer
# to recognize more abbreviations
# https://stackoverflow.com/questions/14095971/how-to-tweak-the-nltk-sentence-tokenizer?
# The list is created by examining the frequency used in
# ~9k random sampled articles in Wikipedia
extra_abbreviations = ['pp', 'no', 'vol', 'ed', 'al', 'e.g', 'etc', 'i.e',
    'pg', 'dr', 'mr', 'mrs', 'ms', 'vs', 'prof', 'inc', 'incl', 'u.s', 'st',
    'trans', 'ex']
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sent_detector._params.abbrev_types.update(extra_abbreviations)

# Wiki markups to be detected for broken sentences:
# reference tags, templates {{}}, wikilinks [[]], parentheses (),
# quotation marks "", italics text ''''
markup_regex = "(<ref)|({{)|(\[\[)|(\()|(</ref>)|(/>)|(}})|(\]\])|(\))|(\")|('{2})"

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
        # Text has wiki markups, need to remove all unprintable
        # code when predicted by the model
        text_stripped = mwparserfromhell.parse(text).strip_code()
        wordlist = text_to_word_list(text_stripped)
        # Construct word vectors
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

def _clean_wikicode(section):
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

def _is_paired(markups):
    return all(((markups['<ref'] == markups['</ref>'] + markups['/>']),
                (markups['{{'] == markups['}}']),
                (markups['[['] == markups[']]']),
                (markups['('] == markups[')']),
                (not markups['"']%2), (not markups['\'\'']%2)))

def _has_opening_markups(sent):
    return any((('<ref' in sent), ('{{' in sent), ('[[' in sent),
                ('(' in sent), ('"' in sent), ('\'\'' in sent)))


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

        _clean_wikicode(section)

        for i, paragraph in enumerate(re.split("\n+", str(section))):
            pid = mkid(section_name+str(i))
            paragraphs[pid] = paragraph

            is_opening = False
            broken_sent = []
            markups = defaultdict(int)
            for sent in sent_detector.tokenize(paragraph.strip()):
                # Detect and fix broken sentences caused by sent_detector
                # not aware of templates and other paired markups.
                # Fixed by joining a broken sentence (e.g., one where
                # has an opening templates with the following one.)
                if is_opening:
                    for match in re.finditer(markup_regex, sent):
                        markups[match.group(0)] += 1
                    broken_sent.append(sent)
                    if _is_paired(markups):
                        sent = ''.join(broken_sent)
                        is_opening = False
                        broken_sent = []
                        markups = defaultdict(int)
                else:
                    if _has_opening_markups(sent):
                        for match in re.finditer(markup_regex, sent):
                            markups[match.group(0)] += 1
                        if not _is_paired(markups):
                            is_opening = True
                            broken_sent.append(sent)
                        else:
                            markups = defaultdict(int)
                if is_opening:
                    continue

                # Skip sentence starts with special characters
                if sent.strip()[0] in "|!<{":
                    continue

                # Sentence has wiki markups, strip all unprintable
                # code to count the number of words correctly and
                # skip sentence shorter than min_sentence_length
                sent_stripped = mwparserfromhell.parse(sent).strip_code()
                if len(word_tokenize(sent_stripped)) < cfg.min_sentence_length:
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


def parse(pageids):
    page_processed = 0
    pageids_list = list(pageids) # for test, sholud upgrade to multi-threads version
    model, vocab_w2v, section_dict = load_citation_needed()

    def insert(cursor, r):
        cursor.execute('''
            INSERT INTO statements (statement, context, section, rev_id, score)
            VALUES(%s, %s, %s, %s, %s)
            ''', r)
    db = cddb.init_scratch_db()

    batch_size = 32
    for i in range(0, len(pageids_list), batch_size):
        rows = []
        results = query_pageids(pageids_list[i: i + batch_size])
        for revid, title, wikitext in results:
            start_len = len(rows)
            wikicode = mwparserfromhell.parse(wikitext)
            sentences, paragraphs = extract(wikicode)

            for text, pidx, section in sentences:
                row = [text, paragraphs[pidx], section, revid]
                rows.append(row)

            pred = run_citation_needed(sentences, model, vocab_w2v, section_dict)
            for i, score in enumerate(pred):
                rows[start_len+i].append(score[1])

            page_processed += 1

        for r in rows:
            try:
                db.execute_with_retry(insert, r)
            except:
                # This may be caused by the size
                # of a sentence or paragraph that
                # exceed the maximum limit
                continue

    print('page processed: ', page_processed)
    return 0

if __name__ == '__main__':
    start = time.time()
    pageids_file = os.path.expanduser('~/citationdetective/pageids')
    with open(pageids_file) as pf:
        pageids = set(map(str.strip, pf))
    ret = parse(pageids)
    print('all done in %d seconds.' % (time.time()-start))
    sys.exit(ret)
