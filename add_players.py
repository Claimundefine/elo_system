import sys  
from peewee import SqliteDatabase, Model, CharField, IntegerField, DateTimeField
from datetime import datetime
from db import db, init_db
from models import Player

elo_map = {
    "iron1": 0,
    "iron2": 100,
    "iron3": 200,
    "bronze1": 300,
    "bronze2": 400,
    "bronze3": 500,
    "silver1": 600,
    "silver2": 700,
    "silver3": 800,
    "gold1": 900,
    "gold2": 1000,
    "gold3": 1100,
    "platinum1": 1200,
    "platinum2": 1300,
    "platinum3": 1400,
    "diamond1": 1500,
    "diamond2": 1600,
    "diamond3": 1700,
    "ascendant1": 1800,
    "ascendant2": 1900,
    "ascendant3": 2000,
    "immortal1": 2100,
    "immortal2": 2200,
    "immortal3": 2300
}

def update_player_win_loss(user, wins, losses):
    if Player.select().where(Player.player == user).exists():
        player = (Player.select()
                        .where(Player.player == user)
                        .order_by(Player.updated_at.desc())
                        .first())
        player.wins = wins
        player.losses = losses
        player.save()
    else:
        raise ValueError(f"Player '{user}' does not exist!")

    for p in Player.select():
        print(f"{p.player} - Wins: {p.wins}, Losses: {p.losses}, Updated at: {p.updated_at}")

def update_player_rank(user, value):
    if Player.select().where(Player.player == user).exists():
        player = (Player.select()
                        .where(Player.player == user)
                        .order_by(Player.updated_at.desc())
                        .first())
        player.rank = value
        player.save()
    else:
        raise ValueError(f"Player '{user}' does not exist!")

    for p in Player.select():
        print(f"{p.player} - {p.rank} - {p.updated_at}")

def add_player_to_db(user, rank):
    if Player.select().where(Player.player == user).exists():
        raise ValueError(f"Player '{user}' already exists!")
    else:
        Player.create(player=user, rank=rank)

    for p in Player.select():
        print(f"{p.player} - {p.rank} - {p.updated_at}")


def main(user, rank, points=0):
    if rank not in elo_map:
        raise ValueError(f"Invalid rank: {rank}. Valid ranks are: {', '.join(elo_map.keys())}")
    
    if not points.isdigit():
        raise ValueError(f"Invalid points: {points}. Points must be a number.")
    points = int(points)
    if points < 0 or points > 100:
        raise ValueError(f"Invalid points: {points}. Points must be between 0 and 100.")
    
    init_db()
    
    add_player_to_db(user, elo_map[rank] + points)
    


if __name__ == "__main__":

    if sys.argv[1] == "update":
        if len(sys.argv) != 4:
            print("Usage: python add_players.py update <user> <rank>")
            sys.exit(1)
        user = sys.argv[2]
        rank = sys.argv[3]
        update_player_rank(user, rank)
        sys.exit(0)
    
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python add_players.py <user> <rank> <points>")
        sys.exit(1)
    
    if len(sys.argv) == 4:
        points = sys.argv[3]
    else:
        points = "0"
    
    main(sys.argv[1], sys.argv[2], points)
