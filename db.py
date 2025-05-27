# db.py or top of your script
from peewee import SqliteDatabase

db = SqliteDatabase('players.db')

def init_db():
    if db.is_closed():
        db.connect()