import config
import utils

import MySQLdb

import contextlib
import os
import time
import warnings

REPLICA_MY_CNF = os.getenv(
    'REPLICA_MY_CNF', os.path.expanduser('~/replica.my.cnf'))
TOOLS_LABS_CH_MYSQL_HOST = 'tools.db.svc.eqiad.wmflabs'

class _RetryingConnection(object):
    '''
    Wraps a MySQLdb connection, handling retries as needed.
    '''

    def __init__(self, connect, sleep = time.sleep):
        self._connect = connect
        self._sleep = sleep
        self._do_connect()

    def _do_connect(self):
        self.conn = self._connect()
        self.conn.ping(True) # set the reconnect flag

    def execute_with_retry(self, operations, *args, **kwds):
        max_retries = 5
        for retry in range(max_retries):
            try:
                with self.conn.cursor() as cursor:
                    return operations(cursor, *args, **kwds)
            except MySQLdb.OperationalError:
                if retry == max_retries - 1:
                    raise
                else:
                    self._sleep(2 ** retry)
                    self._do_connect()
            else:
                break

    def execute_with_retry_s(self, sql, *args):
        def operations(cursor, sql, *args):
            cursor.execute(sql, args)
            if cursor.rowcount > 0:
                return cursor.fetchall()
            return None
        return self.execute_with_retry(operations, sql, *args)

    def __getattr__(self, name):
        return getattr(self.conn, name)

@contextlib.contextmanager
def ignore_warnings():
    warnings.filterwarnings('ignore', category = MySQLdb.Warning)
    yield
    warnings.resetwarnings()

def _connect(**kwds):
    return MySQLdb.connect(charset = 'utf8mb4', autocommit = True, **kwds)

def _connect_to_cd_mysql():
    kwds = {'read_default_file': REPLICA_MY_CNF}
    if utils.running_in_tools_labs():
        kwds['host'] = TOOLS_LABS_CH_MYSQL_HOST
    return _connect(**kwds)

def _connect_to_wp_mysql(cfg):
    kwds = {'read_default_file': REPLICA_MY_CNF}
    if utils.running_in_tools_labs():
        # Get the project database name (and ultimately the database server's
        # hostname) from the name of the database we want, as per:
        # https://wikitech.wikimedia.org/wiki/Help:Tool_Labs/Database#Naming_conventions
        xxwiki = cfg.database.replace('_p', '')
        kwds['host'] = '%s.analytics.db.svc.eqiad.wmflabs' % xxwiki
    return _connect(**kwds)

def _make_tools_labs_dbname(cursor, database, lang_code):
    cursor.execute("SELECT SUBSTRING_INDEX(USER(), '@', 1)")
    user = cursor.fetchone()[0]
    return '%s__%s_%s' % (user, database, lang_code)

def _use(cursor, database, lang_code):
    cursor.execute('USE %s' % _make_tools_labs_dbname(
        cursor, database, lang_code))

def init_db(lang_code):
    def connect_and_initialize():
        db = _connect_to_cd_mysql()
        _use(db.cursor(), 'citationdetective', lang_code)
        return db
    return _RetryingConnection(connect_and_initialize)

def init_scratch_db():
    cfg = config.get_localized_config()
    def connect_and_initialize():
        db = _connect_to_cd_mysql()
        _use(db.cursor(), 'scratch', cfg.lang_code)
        return db
    return _RetryingConnection(connect_and_initialize) 

def init_wp_replica_db(lang_code):
    cfg = config.get_localized_config(lang_code)
    def connect_and_initialize():
        db = _connect_to_wp_mysql(cfg)
        with db.cursor() as cursor:
            cursor.execute('USE ' + cfg.database)
        return db
    return _RetryingConnection(connect_and_initialize)

# Methods for use in batch scripts, not the serving database to tool developers 
# in Toolforge. These set up the databases, help populate the scratch database 
# and swap it with the serving database.

def _create_citationdetective_tables(cfg, cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS statements (
        id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY, hashid VARCHAR(128),
        statement VARCHAR(%s), context VARCHAR(%s), 
        section VARCHAR(768), rev_id INT(8) UNSIGNED, score FLOAT(8))
    ''', (cfg.statement_max_size, cfg.context_max_size))


def initialize_all_databases():
    def _do_create_database(cursor, database, lang_code):
        dbname = _make_tools_labs_dbname(cursor, database, lang_code)
        cursor.execute('SET SESSION sql_mode = ""')
        cursor.execute(
            'CREATE DATABASE IF NOT EXISTS %s '
            'CHARACTER SET utf8mb4' % dbname)
    cfg = config.get_localized_config()
    db = _RetryingConnection(_connect_to_cd_mysql)
    with db.cursor() as cursor, ignore_warnings():
        cursor.execute('DROP DATABASE IF EXISTS ' + _make_tools_labs_dbname(
            cursor, 'scratch', cfg.lang_code))
        for database in ['citationdetective', 'scratch']:
            _do_create_database(cursor, database, cfg.lang_code)
        _use(cursor, 'scratch', cfg.lang_code)
        _create_citationdetective_tables(cfg, cursor)
        _use(cursor, 'citationdetective', cfg.lang_code)
        _create_citationdetective_tables(cfg, cursor)


def install_scratch_db():
    cfg = config.get_localized_config()
    with init_db(cfg.lang_code).cursor() as cursor:
        cdname = _make_tools_labs_dbname(cursor, 'citationdetective', cfg.lang_code)
        scname = _make_tools_labs_dbname(cursor, 'scratch', cfg.lang_code)
        # generate a sql query that will atomically swap tables in
        # 'citationdetective' and 'scratch'. Modified from:
        # http://blog.shlomoid.com/2010/02/emulating-missing-rename-database.html
        cursor.execute('''SET group_concat_max_len = 2048;''')
        cursor.execute('''
            SELECT CONCAT('RENAME TABLE ',
            GROUP_CONCAT('%s.', table_name,
            ' TO ', table_schema, '.old_', table_name, ', ',
            table_schema, '.', table_name, ' TO ', '%s.', table_name),';')
            FROM information_schema.TABLES WHERE table_schema = '%s'
            GROUP BY table_schema;
        ''' % (cdname, cdname, scname))

        rename_stmt = cursor.fetchone()[0]
        cursor.execute(rename_stmt)
        cursor.execute('DROP DATABASE ' + scname)
