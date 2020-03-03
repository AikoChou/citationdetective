#!/usr/bin/env python3

import os
import sys
_upper_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
if _upper_dir not in sys.path:
    sys.path.append(_upper_dir)

import time
import re
import argparse
import multiprocessing
import functools
import traceback
from collections import defaultdict

import cddb
import config
from keras_model import KerasManager
from utils import *

import requests
import mwapi
import mwparserfromhell
import nltk.data
from nltk.tokenize import word_tokenize

cfg = config.get_localized_config()
WIKIPEDIA_BASE_URL = 'https://' + cfg.wikipedia_domain

logger = logging.getLogger('parse')
setup_logger_to_stderr(logger)

manager = KerasManager()
manager.start()
model = manager.KerasModel()

# Tweak the NLTK's pre-trained English sentence tokenizer
# to recognize more abbreviations.
# See https://stackoverflow.com/questions/14095971
# The list is created by examining the frequency used in
# ~9k random sampled articles in Wikipedia.
extra_abbreviations = ['pp', 'no', 'vol', 'ed', 'al', 'e.g', 'etc', 'i.e',
    'pg', 'dr', 'mr', 'mrs', 'ms', 'vs', 'prof', 'inc', 'incl', 'u.s', 'st',
    'trans', 'ex']
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sent_detector._params.abbrev_types.update(extra_abbreviations)

# The wiki markups to be detected for broken sentences:
# reference tags, templates, wikilinks, parentheses, quotation marks, italics text.
# TODO: It would be nice to use the tokenizer from mwparserfromhell,
# to look for a token of type such as TemplateOpen, so we don't need to maintain
# a regular expression list here.
markup_regex = "(<ref)|({{)|(\[\[)|(\()|(</ref>)|(/>)|(}})|(\]\])|(\))|(\")|('{2})"

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

def with_max_exceptions(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwds):
        try:
            return fn(*args, **kwds)
        except:
            traceback.print_exc()
            self.exception_count += 1
            if self.exception_count > MAX_EXCEPTIONS_PER_SUBPROCESS:
                logger.error('Too many exceptions, quitting!')
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
            # Save the text with wiki markups into the database,
            # but we need to remove them when throwing the text
            # to the model for predicting citation needed scores.
            row = [text, paragraphs[pidx], section, revid]
            rows.append(row)
            text = mwparserfromhell.parse(text).strip_code()

        pred = model.run_citation_needed(sentences)
        for i, score in enumerate(pred):
            rows[start_len+i].append(score[1])

    def insert(cursor, r):
        cursor.execute('''
            INSERT INTO sentences (sentence, paragraph, section, rev_id, score)
            VALUES(%s, %s, %s, %s, %s)
            ''', r)
    db = cddb.init_scratch_db()
    for r in rows:
        if r[4] > cfg.min_citation_need_score:
            try:
                db.execute_with_retry(insert, r)
            except:
                continue

def parse(pageids, timeout):
    # Keep the number of processes to cpu_count in the pool
    # for now since no speedup if we spawning more processes.
    # The manager seems to be a bottleneck.
    pool = multiprocessing.Pool()

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
        logger.info('timeout, canceling the process pool!')
        pool.terminate()

    pool.join()
    try:
        result.get()
        ret = 0
    except Exception as e:
        logger.error('Too many exceptions, failed!')
        ret = 1
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pageids_file')
    args = parser.parse_args()
    start = time.time()
    with open(args.pageids_file) as pf:
        pageids = set(map(str.strip, pf))
    logger.info('processing %d articles...' % len(pageids))
    ret = parse(pageids, None)
    logger.info('all done in %d seconds.' % (time.time()-start))
    sys.exit(ret)
