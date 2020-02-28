#!/usr/bin/env python3

import os
import sys
_upper_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
if _upper_dir not in sys.path:
    sys.path.append(_upper_dir)

import cddb

def sanity_check():
    sdb = cddb.init_scratch_db()
    sentence_count = sdb.execute_with_retry_s(
        '''SELECT COUNT(*) FROM sentences''')[0][0]
    assert sentence_count > 100

if __name__ == '__main__':
    sanity_check()
    cddb.install_scratch_db()