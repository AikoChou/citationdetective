#!/usr/bin/env python3

import os
import sys
_upper_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
if _upper_dir not in sys.path:
    sys.path.append(_upper_dir)

import cddb
import config
import utils

import time
import subprocess
import argparse
import tempfile
import dateutil.parser
import datetime
import traceback

def email(message, attachments):
    subprocess.getoutput(
        '/usr/bin/mail -s "%s" ' % message +
        ' '.join('-a ' + a for a in attachments) +
        ' citationdetective.update@tools.wmflabs.org')
    time.sleep(2*60)

def shell(cmdline):
    print('Running %s' % cmdline, file=sys.stderr)
    status, output = subprocess.getstatusoutput(cmdline)
    print(output, file=sys.stderr)
    return status == 0

def _update_db_tools_labs(cfg):
    os.environ['CD_LANG'] = cfg.lang_code
    cddb.initialize_all_databases()

    # FIXME Import and calll these scripts instead of shelling out?
    def run_script(script, cmdline = '', optional = False):
        scripts_dir = os.path.dirname(os.path.realpath(__file__))
        script_path = os.path.join(scripts_dir, script)
        cmdline = ' '.join([sys.executable, script_path, cmdline])
        assert shell(cmdline) == True or optional, 'Failed at %s' % script

    pageids = tempfile.NamedTemporaryFile()
    run_script(
        'print_pageids_from_wikipedia.py', '> ' + pageids.name)
    run_script('parse.py', pageids.name)
    run_script('install_new_database.py')

    pageids.close()  # deletes the file

def update_db_tools_labs(cfg):
    # Should match the job's name in crontab
    logfiles = [
        'cd_update_' + cfg.lang_code + '.' + ext
        for ext in ('out', 'err')
    ]
    for logfile in logfiles:
        open(logfile, 'w').close()  # truncate

    try:
        email('Start to build database', logfiles)
        _update_db_tools_labs(cfg)
        email('Successful to build database', logfiles)
    except Exception as e:
        traceback.print_exc(file = sys.stderr)
        email('Failed to build database', logfiles)
        sys.exit(1)
    utils.mkdir_p(cfg.log_dir)
    for logfile in logfiles:
        os.rename(logfile, os.path.join(cfg.log_dir, logfile))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Update the Citation Detective databases.')
    parser.add_argument('lang_code',
        help='One of the language codes in ../config.py')
    args = parser.parse_args()

    if not utils.running_in_tools_labs():
        print('Not running in Tools Labs!', file=sys.stderr)
        sys.exit(1)

    if args.lang_code not in config.LANG_CODES_TO_LANG_NAMES:
        print('Invalid lang code! Use one of: ', end=' ', file=sys.stderr)
        print(list(config.LANG_CODES_TO_LANG_NAMES.keys()), file=sys.stderr)
        parser.print_usage()
        sys.exit(1)

    cfg = config.get_localized_config(args.lang_code)
    update_db_tools_labs(cfg)
