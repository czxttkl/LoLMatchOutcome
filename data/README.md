# Raw Data Import

Most training data sets are stored as pickle files stored in `input` folder. 
These pickle files are generated from scripts in `data_collect` folder. 
These scripts generate required pickle files from a mongodb where raw match data is stored.
This `README` file describes the procedure to set up the mongodb.

1. Download mongodb community version according to https://docs.mongodb.com/manual/tutorial/install-mongodb-on-linux/, 
unzip it and run it by 
```
bin/mongod --dbpath <data_path>
```
where `data_path` is a self-defined directory for storing mongodb data.

2. Download mongodb dump file: https://www.dropbox.com/s/2mnf0gv00zarjje/dump.agz?dl=0 

3. use mongodump to restore the database
```
bin/mongorestore --drop --gzip --archive=dump.agz
```
80GB free disk space is needed.

4. connect to mongodb shell and check the database is successfully restored:
```
bin/mongo
> use lol
switched to db lol
> show collections
match
match_seed
player_seed
player_seed_match_history
```

The four collections are:
* `match_seed`: 205,573 matches which take place in region NA, by players with tier platinum or above, happening in the first 10 days since version 8.6 was released, and restricted to queue 420 (i.e., Summoner's Rift, 5v5 Ranked Solo games)
* `player_seed`: 193,632 players who appear in any of the matches in `match_seed`
* `player_seed_match_history`: match history of the players in `player_seed`, each <match_id, player_id> takes up one entry in this collection
* `match`: detailed statistics of each match in match history. While `player_seed_match_history` contains non queue 420 matches, so far we have only crawled only queue 420 match details in `match`

The verbose procedure for crawling each collection is noted here: https://github.com/czxttkl/GAE/blob/master/data_collect/README.md  


 

 
