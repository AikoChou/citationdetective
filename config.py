import os
import types
from functools import reduce

_GLOBAL_CONFIG = dict(
    # If running on Tools labs, keep database dumps in this directory...
    archive_dir = os.path.join(os.path.expanduser('~'), 'archives'),

    # ...and delete dumps that are older than this many days
    archive_duration_days = 90,

    # Where to put various logs
    log_dir = os.path.join(os.path.expanduser('~'), 'logs'),

    flagged_off = [],

    profile = True,

    stats_max_age_days = 90,

    user_agent = 'citationdetective',

)

_CITATION_NEEDED_CONFIG = dict(
    # Embeddings for the words in the sentences
    vocb_path = os.path.expanduser('~/citation-needed/embeddings/word_dict_en.pck'),

    # Embeddings for the section titles
    section_path = os.path.expanduser('~/citation-needed/embeddings/section_dict_en.pck'),

    # Tensorflow models to detect Citation Need for English
    model_path = os.path.expanduser('~/citation-needed/models/fa_en_model_rnn_attention_section.h5'),

    # Maximum length of all sequences
    max_seq_length = 187,

)

# A base configuration that all languages "inherit" from.
_BASE_LANG_CONFIG = dict(

    statement_max_size = 200,

    context_max_size = 800,
    
    hyperlink_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",

    sentence_regex = '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',

)

# Language-specific config, inheriting from the base config above.
_LANG_CODE_TO_CONFIG = dict(
    en = dict(
        # A friendly name for the language
        lang_name = 'English',

        # The direction of the language, either ltr or rtl
        lang_dir = 'ltr',
        
        # The database to use on Tools Labs
        database = 'enwiki_p',
        
        # The domain for Wikipedia in this language
        wikipedia_domain = 'en.wikipedia.org',
        
        # These sections which content do not need citations
        sections_to_skip = [
            'See also', 
            'References',
            'External links',
            'Further reading',
            'Notes',
        ],
        
        # https://en.wikipedia.org/wiki/Special:Statistics
        # As of 31 December 2019, there are 5,989,239 articles in the English Wikipedia.
        num_pages = 6000000,

        # X% of all articles
        X = 1, 
    ),
)

Config = types.SimpleNamespace

def _inherit(base, child):
    ret = dict(base)  # shallow copy
    for k, v in child.items():
        if k in ret:
            if isinstance(v, list):
                v = ret[k] + v
            elif isinstance(v, dict):
                v = dict(ret[k], **v)
        ret[k] = v
    return ret

LANG_CODES_TO_LANG_NAMES = {
    lang_code: _LANG_CODE_TO_CONFIG[lang_code]['lang_name']
    for lang_code in _LANG_CODE_TO_CONFIG
}

def get_localized_config(lang_code='en'):
    if lang_code is None:
        lang_code = os.getenv('CITATION_LANG')
    lang_config = _LANG_CODE_TO_CONFIG[lang_code]
    cfg = Config(lang_code = lang_code, **reduce(
        _inherit, [_GLOBAL_CONFIG, _CITATION_NEEDED_CONFIG, _BASE_LANG_CONFIG, lang_config]))
    cfg.lang_codes_to_lang_names = LANG_CODES_TO_LANG_NAMES
    return cfg
