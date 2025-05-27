from peewee import Model, CharField, FloatField, DateTimeField
from datetime import datetime
from db import db  # import the shared db instance

class BaseModel(Model):
    class Meta:
        database = db

class Player(BaseModel):
    player = CharField()
    rank = FloatField()
    updated_at = DateTimeField(default=datetime.now)

class Game(BaseModel):
    filename = CharField(unique=True)  # image file name, e.g. "match_001.png"
    processed_at = DateTimeField(default=datetime.now)

class MatchHistory(BaseModel):
    player = CharField()
    filename = CharField()  # links to Game.filename
    kills = FloatField()
    deaths = FloatField()
    assists = FloatField()
    first_bloods = FloatField()
    result = CharField()  # e.g., "win" or "loss"
    created_at = DateTimeField(default=datetime.now)