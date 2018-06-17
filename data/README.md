# Raw Data Import

Most training data sets are stored as pickle files stored in `input` folder. 
These pickle files are generated from scripts in `data_collect` folder. 
These scripts generate required pickle files from a mongodb where raw match data is stored.
This README file describes the procedure to set up the mongodb.

1. Download mongodb community version according to https://docs.mongodb.com/manual/tutorial/install-mongodb-on-linux/, 
unzip it and run it by 
```
bin/mongod --dbpath <data_path>
```
2. use mongodump to restore the database.



 

 
