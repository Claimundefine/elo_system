from peewee import SqliteDatabase, Model, CharField, IntegerField, DateTimeField, FloatField
from datetime import datetime
from models import Player, Game, MatchHistory
from db import init_db
from tabulate import tabulate


init_db()  # Ensure the database is initialized


player_map = {}

unique_players = Player.select(Player.player).distinct()
player_list = [player.player for player in unique_players]
for player in player_list:
    query = (Player.select(Player.player, Player.rank, Player.updated_at)
             .where(Player.player == player)
                    .order_by(Player.updated_at.desc())
                    .limit(2))
    player_map[player] = []
    if len(query) == 2:
        player_map[player].append([query[0].rank, query[1].rank])
    elif len(query) == 1:
        player_map[player].append([query[0].rank])

    match_query = (MatchHistory.select(MatchHistory.result)
                   .where(MatchHistory.player == player))
    
    wins, losses = 0, 0

    for match in match_query:
        if match.result == "win":
            wins += 1
        elif match.result == "loss":
            losses += 1

    player_map[player].append((wins, losses))

final_list = [[key] + [player_map[key][0], player_map[key][1][0], player_map[key][1][1]] for key in player_map]
final_list.sort(key=lambda x: x[1][0], reverse=True)

formatted = []
for name, elos, wins, losses in final_list:
    if len(elos) > 1:
        elo_diff = round(elos[0] - elos[1], 2)
    else:
        elo_diff = 0
    formatted.append([name, elos[0], elo_diff, wins, losses])

print(tabulate(formatted, headers=["Player", "Current ELO", "Δ ELO", "Wins", "Losses"]))

