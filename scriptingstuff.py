from peewee import SqliteDatabase, Model, CharField, IntegerField, DateTimeField, FloatField
from datetime import datetime
from models import Game, MatchHistory, Player
from db import db, init_db  # import the shared db instance
from playhouse.migrate import SqliteMigrator, migrate

init_db()  # Ensure the database is initialized

query = Player.delete().where(Player.player == "wint")
query.execute()
