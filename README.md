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

### Access to the database in Toolforge
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
