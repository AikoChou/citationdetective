import os
import re
import hashlib
import logging.handlers

def e(s):
    if type(s) == bytes:
        return s
    return s.encode('utf-8')

def d(s):
    if type(s) == str:
        return s
    return str(s, 'utf-8')

def mkid(s):
    return hashlib.sha1(e(s)).hexdigest()[:2*4]

def running_in_tools_labs():
    return (os.path.exists('/etc/wmflabs-project') or
        os.path.exists('/etc/wmcs-project'))

# Thanks, StackOverflow! https://stackoverflow.com/questions/600268
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def text_to_word_list(text):
    text = str(text).lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"-", " ", text) #
    text = re.sub(r";", " ", text) #
    text = re.sub(r":", " ", text) #
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()
    return text

def _setup_log_handler(logger, handler):
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [%(pathname)s:%(lineno)d]'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def setup_logger_to_stderr(logger):
    _setup_log_handler(logger, logging.StreamHandler())
