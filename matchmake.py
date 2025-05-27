import sys
from db import db, init_db
from models import Player
from main import check_all_players_exist, all_current_elos, team_elo, skew_relative_team_elo, expected_score
import itertools


def generate_team_splits(players):
    n = len(players)
    if n % 2 != 0:
        raise ValueError("Number of players must be even.")

    half = n // 2
    seen = set()
    teams = []

    for combo in itertools.combinations(players, half):
        team_a = set(combo)
        team_b = tuple(sorted(set(players) - team_a))

        key = tuple(sorted(team_a))  # avoid mirrored duplicates
        if key not in seen:
            seen.add(key)
            teams.append((sorted(team_a), list(team_b)))

    return teams


def matchmake(teams, player_hashmap):

    teamAList = None
    teamBList = None
    min_diff = float('inf')
    team1_aggregate = 0
    team2_aggregate = 0

    for team in teams:
        team1, team2 = team
        team1_elo = team_elo(team1, player_hashmap)
        team2_elo = team_elo(team2, player_hashmap)
        teamA, teamB, _, _ = skew_relative_team_elo(team1_elo, team2_elo)
        diff = abs(teamA - teamB)
        if diff < min_diff:
            min_diff = diff
            teamAList = team1
            teamBList = team2 
            team1_aggregate = teamA
            team2_aggregate = teamB

    print(f"Team 1 aggregate ELO: {team1_aggregate}")
    print(f"Team 2 aggregate ELO: {team2_aggregate}")
    
    print(f"Expected winrate for team 1: {expected_score(team1_aggregate, team2_aggregate)}")
    print(f"Expected winrate for team 2: {expected_score(team2_aggregate, team1_aggregate)}")

    return teamAList, teamBList


def main(names):
    init_db()

    if not check_all_players_exist(names):
        print("One or more players do not exist in the database.")
        sys.exit(1)

    print("All players exist in the database. Proceeding with matchmaking...")
    player_hashmap = {}
    all_current_elos(player_hashmap, names)
    print("Current ELOs:", player_hashmap)
    teams = generate_team_splits(names)
    balanced_teams = matchmake(teams, player_hashmap)
    print("Matchmaking complete!")
    print(f"Team A: {balanced_teams[0]}")
    print(f"Team B: {balanced_teams[1]}")
        

if __name__ == "__main__":
    print("Matchmaking script started.")
    if len(sys.argv) != 2:
        print("Usage: python matchmake.py name1,name2,...,nameN")
        sys.exit(1)

    names = sys.argv[1].split(",")
    if not names or len(names) != 10:
        print("Please provide exactly 10 names separated by commas.")
        sys.exit(1)

    print(f"Names provided: {names}")

    main(names)
    db.close()  # Close the database connection when done
    print("Database connection closed.")