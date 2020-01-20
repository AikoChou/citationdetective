import re
from collections import namedtuple
from types import SimpleNamespace
from nltk.tokenize import sent_tokenize

Citation = namedtuple('Citation', ['text', 'prev', 'start', 'end'])

class CitationParser(object):

    def __init__(self):
        self._ref_regex = '<ref.+?(/>|</ref>)'
        self._cn_regex = '{{(C|c)itation needed\|.+?}}'
        self._citation_regex = self._ref_regex + '|' + self._cn_regex
        self._citations = []
        self._sentences = []

    def _find_citations(self, wikitext):
        for ref in re.finditer(self._citation_regex, wikitext):
            self._citations.append(Citation(text=ref.group(0), prev=wikitext[ref.start(0)-1], 
                start=ref.start(0), end=ref.start(0)+len(ref.group(0))))
        self._citations.reverse()

    def _remove_citation_tags(self, wikitext):
        return re.sub(self._citation_regex, '', wikitext)

    def _add_successive_citation(self):
        while self._citations:
            if self._citations[-1].prev == '>':
                self._sentences[-1].cite.append(self._citations.pop())
            else:
                break

    def parse(self, wikitext):
        """
        This is a method for parsing wikitext to get individual sentences
        and the citations they have.
        
        The algorithm goes like:
        1) Find reference tags <ref> and citation needed templates {cn}
           within the wikitext and store them in a queue in context order.
        2) Get the plain text by removing <ref> and {cn}.
        3) Split the plain text into sentences.
        4) Loop over the sentences to see whether we can find the sentence
           in the wikitext.
           -If found, that means the sentence isn't broken by <ref> or
            {cn}, then we check whether any citations is followed behind
            the sentence.
           -If not found, the sentence is broken by <ref> or {cn}, we find
            the correct start location and end location of the sentence
            and also check whether any citation is followed behind.

        """
        self._find_citations(wikitext)
        plain_text = self._remove_citation_tags(wikitext)
        for i, sent in enumerate(sent_tokenize(plain_text)):
            start = wikitext.find(sent)
            if start >= 0:
                # We find the complete sentence in the wikitext, just check if 
                # any citations is followed.
                if self._citations and self._citations[-1].start == start+len(sent):
                    # If a citation is attached behind, the start location
                    # is equal to the end location of the sentence.
                    self._sentences.append(SimpleNamespace(text=wikitext[start:start+len(sent)], 
                        start=start, end=start+len(sent), cite=[self._citations.pop()]))
                    self._add_successive_citation()
                else:
                    self._sentences.append(SimpleNamespace(text=wikitext[start:start+len(sent)], 
                        start=start, end=start+len(sent), cite=None))
            else:
                # We can't find the complete sentence in the wikitext.
                # First we need to identify the correct start location
                # of the sentence.
                if i == 0:
                    # The sentence is the first sentence in the wikitext.
                    start = 0
                else:
                    # The sentence is either followed by another sentence
                    # or a citaion.
                    if self._sentences[-1].cite:
                        start = max(self._sentences[-1].end+1, self._sentences[-1].cite[-1].end+1)
                    else:
                        start = self._sentences[-1].end+1

                self._sentences.append(SimpleNamespace(text='',
                    start=start, end=start+len(sent), cite=[self._citations.pop()]))
                self._add_successive_citation()

                # Identify the end location of the sentence. Since the citation is in the
                # middle of the sentence, we need to take into account the length of the
                # citation itself.
                self._sentences[-1].end += sum([len(c.text) for c in self._sentences[-1].cite])
                self._sentences[-1].text = wikitext[start:self._sentences[-1].end]

                # Check whether any citation is followed behind.
                if self._citations and self._citations[-1].start == self._sentences[-1].end:
                    self._sentences[-1].cite.append(self._citations.pop())
                    self._add_successive_citation()

        # Ideally, we expect an empty citation queue
        # after we pass over every sentence. If not
        # the case, we empty it manually.
        if self._citations:
            self._citations = []

    def clear_sentences(self):
        self._sentences = []

    def filter_citations(self):
        citations = []
        for sentence in self._sentences:
            if sentence.cite:
                citations.extend(sentence.cite)
        return citations

    def filter_unsourced(self):
        unsourced_sentences = []
        for sentence in self._sentences:
            if sentence.cite:
                cited = [True for c in sentence.cite if re.match(self._ref_regex, c.text)]
                if any(cited):
                    continue
            unsourced_sentences.append(sentence.text)
        return unsourced_sentences

    def print_sentences(self):
        for sentence in self._sentences:
            print('<Sentence span=({}, {}) {}'.format(sentence.start, sentence.end, sentence.text))
            if sentence.cite:
                for c in sentence.cite:
                    print('    <Citation span=({}, {}) {}'.format(c.start, c.end, c.text))
