# Citation Detective

Citation Detective is a system that applies the Citation Need model, a machine-learning-based classifier [published in WWW'19](https://arxiv.org/pdf/1902.11116.pdf) by WMF researchers and collaborators, to a large number of articles in English Wikipedia, producing a dataset that contains sentences detected as missing citations with their associated metadata.

Citation Detective database is now available on the [Wikimedia Toolforge](https://tools.wmflabs.org) as the public SQL database `s54245__citationdetective_p`.  

Every time we update the database, Citation Detective takes randomly about 120k articles in English Wikipedia, runs the Citation Need model for predicting a score that a sentence needs a citation. Citation Detective then extracts sentences with a score higher than 𝑦ˆ >= 0.5 along with contextual information, resulting in hundreds thousand sentences in the database which are classified as needing citations.

A design specification for the system can be found in [this blog post](https://rollingmist.home.blog/2019/12/20/citation-detective-design-specification/) and more information in our [Wiki Workshop submission](https://commons.wikimedia.org/wiki/File:Citation_Detective_WikiWorkshop2020.pdf).

Schema of the *Sentences* table in Citation Detective database:

| Field | Type | Description |
| --- | --- | --- |
| id | integer | Primary key |
| sentence | string | The text of the sentence |
| paragraph | string | The text of the paragraph which contains the sentence | 
| section | string | The section title |
| rev_id | integer | The revision ID of the article |
| score | float | The predicted citation need score |

## Access to the database in Toolforge
You need a developer account to access the database, create one and setup a SSH key refer to the instructions [here](https://wikitech.wikimedia.org/wiki/Portal:Toolforge/Quickstart).

After logging in to Toolforge server, connect to tools.db.svc.eqiad.wmflabs with the replica.my.cnf credentials:
```
$ mysql --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.eqiad.wmflabs
```
You could also just type:
```
$ sql tools
```
Access to Citation Detective database:
```
MariaDB [(none)]> use s54245__citationdetective_p;
```
Access to the database from outside the Toolforge environment is not currently possible, but is under investigation for the future.

## Deploying in Toolforge
*If you want to contribute to this project, please take a look the following instructions for deploying in Toolforge and running locally for development. :slightly_smiling_face:*

The job Citation Detective updates its database runs on the grid engine via Cron.

After logging in to Toolforge server, create a virtualenv and activate it:
```
$ mkdir www/python/
$ virtualenv --python python3 www/python/venv/
$ . www/python/venv/bin/activate
```
Next, clone this repository and install the dependencies:
```
$ git clone https://github.com/AikoChou/citationdetective.git
$ pip install -r citationdetective/requirements.txt
```
Then, download the Citation Need models and embeddings following the instructions in `citation-needed/` directory and put them into corresponded folders. File structure is like:
```
.
└── citation-needed/
    └── embeddings/
        └── word_dict_en.pck
        └── section_dict_en.pck
    └── models/
        └── fa_en_model_rnn_attention_section.h5    
```
Finaly, the `scripts/update_db_tools_labs.py` script automates the generation of the database in Toolforge. It is run regularly as a cron job and needs to run from a virtualenv.
```
/usr/bin/jsub -mem 10g -N cd_update_en -once \
  /data/project/citationdetective/www/python/venv/bin/python3 \
  /data/project/citationdetective/citationdetective/scripts/update_db_tools_labs.py en
```

## Generating the database locally
To generate your own Citation Detective database locally, you need a local installation of MySQL.

First, set a MySQL config file to let the scripts know how to find and log in to the databases: (like the MySQL credentials in ~/replica.my.cnf in Toolforge)
```
$ cat ~/replica.my.cnf
[client]
user='root'
host='localhost'
```
Citation Need model exist for English, Italian and French, and they can be retrained for any language. The scripts expect an environment variable `CD_LANG` to be set as a language code taken from config.py.

Since Citation Detective now only support Englich Wikipedia, we set the variable to `en`:
```
$ export CD_LANG=en
```
Now, let's create all necessary databases and tables:
```
$ python -c 'import cddb; cddb.initialize_all_databases()'
```
Change to `scripts/` directory, run the `parse.py` script which will query the Wikipedia API for the actual content of the pages and run Citation Need model to identify sentences lacking citations:
```
$ cd scripts
$ python parse.py sample_pageids
```
You can use the `sample_pageids` provided or generate one from `print_pageids_from_wikipedia.py`. For the later option, you need to download the page SQL dump of Wikipedia and import the dump in your local MySQL in advance.

Lastly, your MySQL installation should contain a database named `root__scratch_en` with *sentences* table. The `install_new_database.py` script will atomically move the table to a new database named `root__citationdetective_p` which serves as the final database.
```
$ python install_new_database.py
```
