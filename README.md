# Citation Detective

Citation Detective is a system that applies the Citation Need models to a large number of articles in English Wikipedia, producing a dataset which contains sentences detected as missing citations with their associated metadata. (See the database schema below)

Citation Detective database is now available on the Wikimedia Toolforge as the public SQL database `s54245__citationdetective_p`.  

Every time we update the database, Citation Detective takes **randomly about 100k articles** in English Wikipedia, resulting in hundreads thousand sentences in the database which are classified as needing citations. 

Schema of the *Sentences* table in Citation Detective database:

| Field | Type | Description |
| --- | --- | --- |
| id | integer | Primary key |
| sentence | string | The text of the sentence |
| paragraph | string | The text of the paragraph which contains the sentence | 
| section | string | The section title |
| rev_id | integer | The revision ID of the article |
| score | float | The predicted citation need score |

Access to the database from outside the Toolforge environment is not currently possible, but is under investigation for the future.
