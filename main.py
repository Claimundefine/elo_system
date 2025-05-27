import cv2
import easyocr
import pytesseract
import numpy as np
import glob
from peewee import fn
from datetime import datetime
from db import db, init_db
from models import Player, Game, MatchHistory
import math
import sys

def draw_boxes(filename):
    image = cv2.imread(filename)
    clone = image.copy()

    # Initialize state
    drawing = False
    start_point = (-1, -1)
    rectangles = []

# Mouse callback function
    def draw_rectangle(event, x, y, flags, param):
        global start_point, drawing, image

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                image = clone.copy()
                cv2.rectangle(image, start_point, (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            rectangles.append((start_point, end_point))

# Set up window
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

# Main loop
    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()

# Output your rectangles (optional)
    print("Selected Rectangles:")
    for i, rect in enumerate(rectangles):
        print(f"Box {i+1}: {rect}")

def is_seven(region, template_path="template/seven.png", threshold=0.9):
    """
    region: cropped image (from your bounding box)
    template_path: path to a clean '7' template image
    threshold: match confidence threshold
    Returns: True if it's a 7, False otherwise
    """
    template = cv2.imread(template_path, 0)  # grayscale
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        region_gray = region.copy()
    
    # Resize to match template size if needed
    if region_gray.shape != template.shape:
        region_gray = cv2.resize(region_gray, (template.shape[1], template.shape[0]))

    # Match
    res = cv2.matchTemplate(region_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    return max_val >= threshold

def is_two(region, template_path="template/two.png", threshold=0.9):
    """
    region: cropped image (from your bounding box)
    template_path: path to a clean '2' template image
    threshold: match confidence threshold
    Returns: True if it's a 2, False otherwise
    """
    template = cv2.imread(template_path, 0)  # grayscale
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        region_gray = region.copy()
    
    # Resize to match template size if needed
    if region_gray.shape != template.shape:
        region_gray = cv2.resize(region_gray, (template.shape[1], template.shape[0]))

    # Match
    res = cv2.matchTemplate(region_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    return max_val >= threshold

def detect_gamescore(filename):
    reader = easyocr.Reader(['en'])
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filename}")
    print(f"[DEBUG] Image shape: {img.shape}")

    score_bboxes = [[739, 813, 84, 160], [1053, 1118, 90, 158]]
    result_bbox = [[840, 1025, 85, 161]]

    def crop_and_preprocess(box):
        x1, x2, y1, y2 = box
        region = img[y1:y2, x1:x2]

        region = cv2.resize(region, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    scores = []
    for box in score_bboxes:
        if is_seven(crop_and_preprocess(box)):
            scores.append("7")
            print("Detected 7")
        elif is_two(crop_and_preprocess(box)):
            scores.append("2")
            print("Detected 2")
        else:
            region = crop_and_preprocess(box)
            result = reader.readtext(region, detail=0, allowlist="0123456789")
            scores.append(result[0] if result else "?")
    result_box = crop_and_preprocess(result_bbox[0])
    result_text = reader.readtext(result_box, detail=0)
    result_final = result_text[0] if result_text else "?"


    return scores[0], scores[1], result_final


def classify_color(bgr):
    # Custom manual color overrides
    manual_colors = {
        "green": (116, 130, 27),
        "GREEN": (81, 112, 118),
    }

    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    # Check manual mappings
    for name, ref_bgr in manual_colors.items():
        if color_distance(bgr, ref_bgr) < 30:  # tolerance for "closeness"
            return name

    # Fallback to HSV classification
    color = np.uint8([[bgr]])
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    if 30 < h < 50 and s > 100 and v > 100:
        return "GREEN"
    elif h < 10 or h > 160:
        return "red"
    elif 35 < h < 85:
        return "green"
    else:
        return "unknown"
    

def detect_team(filename):
    image = cv2.imread(filename)

    boxes = [
        ((494, 353), (599, 382)),
        ((488, 404), (598, 433)),
        ((484, 457), (583, 486)),
        ((480, 505), (588, 534)),
        ((491, 558), (600, 585)),
        ((491, 615), (603, 645)),
        ((497, 665), (608, 696)),
        ((478, 712), (607, 744)),
        ((479, 767), (589, 795)),
        ((484, 817), (589, 848)),
    ]

    colors = []
    for ((x1, y1), (x2, y2)) in boxes:
        roi = image[y1:y2, x1:x2]

        avg_color = cv2.mean(roi)[:3]
        avg_bgr = tuple(map(int, avg_color))

        label = classify_color(avg_bgr)

        colors.append(label.lower())

    return colors


def detect_scorelines(filename, green_win):
    image = cv2.imread(filename)
    image = cv2.resize(image, (0, 0), fx=3, fy=3)
    all_boxes = {
    "username": [((334, 353), (527, 386)), ((335, 402), (508, 438)), ((334, 456), (500, 488)), ((331, 511), (512, 541)),
             ((331, 561), (516, 592)), ((336, 612), (504, 642)), ((338, 663), (480, 695)), ((327, 714), (483, 750)),
             ((331, 765), (452, 801)), ((329, 820), (485, 851))],
    "combat_score": [((693, 351), (769, 384)), ((695, 407), (776, 437)), ((696, 456), (773, 488)), ((695, 510), (769, 541)),
             ((696, 561), (766, 592)), ((698, 615), (769, 645)), ((694, 666), (768, 698)), ((700, 721), (764, 750)),
             ((700, 767), (763, 798)), ((699, 816), (765, 851))],
    "kill": [((836, 356), (868, 382)), ((832, 405), (871, 433)), ((835, 460), (870, 486)), ((835, 511), (869, 537)),
             ((833, 561), (867, 591)), ((836, 618), (869, 641)), ((839, 663), (869, 696)), ((838, 716), (867, 746)),
             ((838, 764), (870, 799)), ((838, 817), (868, 853))],
    "death": [((888, 358), (918, 381)), ((891, 408), (918, 433)), ((889, 460), (919, 486)), ((889, 510), (918, 537)),
             ((888, 560), (918, 589)), ((893, 615), (917, 641)), ((889, 667), (918, 693)), ((887, 717), (917, 745)),
             ((887, 771), (917, 797)), ((887, 823), (918, 849))],
    "assist": [((938, 358), (966, 381)), ((938, 407), (967, 432)), ((937, 455), (966, 483)), ((936, 511), (968, 539)),
             ((939, 561), (968, 591)), ((937, 618), (967, 641)), ((940, 666), (968, 694)), ((936, 720), (969, 747)),
             ((941, 770), (967, 796)), ((941, 823), (970, 851))],
    "first_blood": [((1180, 352), (1223, 385)), ((1184, 402), (1221, 437)), ((1181, 454), (1223, 491)), ((1182, 508), (1223, 542)),
             ((1186, 561), (1217, 591)), ((1183, 611), (1221, 645)), ((1185, 666), (1220, 697)), ((1183, 714), (1221, 745)),
             ((1187, 768), (1219, 799)), ((1184, 821), (1222, 852))],
    }

    # Multiply all bounding box coordinates by 3
    for key in all_boxes:
        all_boxes[key] = [((x1 * 3, y1 * 3), (x2 * 3, y2 * 3)) for ((x1, y1), (x2, y2)) in all_boxes[key]]

    columns = [all_boxes[key] for key in all_boxes]

    # OCR config
    text_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:.,'
    number_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

    rows = []

    # Assume all columns have the same number of rows
    num_rows = len(columns[0])

    for row_idx in range(num_rows):
        row = []
        for i in range(len(columns)):
            (x1, y1), (x2, y2) = columns[i][row_idx]
            roi = image[y1:y2, x1:x2]

            # Preprocess
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # OCR
            if i == 0:
                text = pytesseract.image_to_string(thresh, config=text_config).strip()
            else:
                text = pytesseract.image_to_string(thresh, config=number_config).strip()
            row.append(text)
    
        rows.append(row)

    teams = detect_team(filename)
    for i, team in enumerate(teams):
        rows[i].append(team)

    rows = sorted(rows, key=lambda x: x[6])  # Sort by team

    #Always return the winning team first
    if green_win:
        return rows[:5], rows[5:]

    return rows[5:], rows[:5]


def detect_files():
    image_files = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg")
    unprocessed_files = [f for f in image_files if not Game.select().where(Game.filename == f).exists()]

    return unprocessed_files
    # return image_files

def check_all_players_exist(player_list):
    for player in player_list:
        player_entry = (Player
                        .select()   
                        .where(Player.player == player)
                        .first())
        if not player_entry:
            print(f"Player {player} does not exist in the database.")
            sys.exit(1)
    return True


def add_to_match_history(team, filename, result):
    # print(f"Adding match history for {filename} with result {result}")
    # print(f"Team: {team}")
    for player in team:
        if not Player.select().where(Player.player == player[0]).exists():
            raise ValueError(f"Player '{player}' does not exist in the database!")
    
        if MatchHistory.select().where((MatchHistory.player == player[0]) & (MatchHistory.filename == filename)).exists():
            print(f"Match history for {player} in {filename} already exists. Skipping.")
            continue
    
        MatchHistory.create(player=player[0], filename=filename, kills=player[2], deaths=player[3],
                        assists=player[4], first_bloods=player[5], result=result)
        print(f"Added match history for {player} in {filename}.")
        

def all_current_elos(hashmap, all_players):
    for player in all_players:
        player = player.replace(" ", "")
        player_entry = (Player
                        .select()
                        .where(Player.player == player)
                        .order_by(Player.updated_at.desc())
                        .first())
        if player_entry:
            print(f"{player} - {player_entry.rank} - {player_entry.updated_at}")
            hashmap[player] = [player_entry.rank, player_entry.updated_at]
        else:
            print(f"Player {player} does not exist in the database.")

def team_elo(team, hashmap):
    team_elo = []
    for player in team:
        if player in hashmap:
            team_elo.append(hashmap[player][0])
        else:
            print(f"Player {player} does not exist in the database.")
    return team_elo

def points_for_even(green_score, red_score, uncertainty=1.5):
    return uncertainty * (17 + (max(green_score, red_score) - 
                                min(green_score, red_score)) * (18 / 13))

def skew_relative_team_elo(team_A, team_B, alpha=1.5):
    full_lobby = team_A + team_B
    avg_lobby_elo = sum(full_lobby) / len(full_lobby)

    def relative_weights(team):
        # Step 1–3: compute skewed relative weights
        deltas = [elo - avg_lobby_elo for elo in team]
        skewed = [math.copysign(abs(d)**alpha, d) for d in deltas]

        # Step 4: shift all weights to positive (e.g., add offset)
        min_w = min(skewed)
        offset = -min_w if min_w < 0 else 0
        positive = [w + offset + 1e-6 for w in skewed]  # avoid 0

        # Step 5: normalize within team
        total = sum(positive)
        return [w / total for w in positive]

    weights_A = relative_weights(team_A)
    weights_B = relative_weights(team_B)

    skewed_A = sum(r * w for r, w in zip(team_A, weights_A))
    skewed_B = sum(r * w for r, w in zip(team_B, weights_B))

    return skewed_A, skewed_B, weights_A, weights_B


def expected_score(r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

def elo_change(player_elo, opponent_elo, result, k):
    """
    result: 1 = win, 0 = loss
    returns: ΔELO (positive for gain, negative for loss)
    """
    e = expected_score(player_elo, opponent_elo)
    return round(k * (result - e), 2)


def normalize_stat(value, team_values):
    min_val, max_val = min(team_values), max(team_values)
    if max_val == min_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)


def compute_team_elo_deltas_with_floor(team_stats, team_delta,
                                       weight_ka=0.3, weight_deaths=0.1, weight_first=0.2,
                                       floor_percent=0.1, elo_weight=0.5):
    """
    For a WIN: distributes a fixed total ELO gain (team_delta > 0)
    Low-ELO players gain more for overperforming; high-ELO players gain less if underperforming.
    """
    assert team_delta > 0, "team_delta must be positive for wins"
    n = len(team_stats)
    floor_total = team_delta * floor_percent
    remaining_delta = team_delta - floor_total
    floor_per_player = round(floor_total / n, 6)

    # Prepare stat vectors
    kas = [p["kills"] * 1.5 + p["assists"] * 0.5 for p in team_stats]
    deaths = [p["deaths"] for p in team_stats]
    firsts = [p["first_bloods"] for p in team_stats]
    elos = [p["elo"] for p in team_stats]
    avg_elo = sum(elos) / n

    raw_scores = []
    for i, p in enumerate(team_stats):
        # Normalized performance
        norm_ka = normalize_stat(kas[i], kas)
        norm_deaths = 1 - normalize_stat(p["deaths"], deaths)
        norm_first = normalize_stat(p["first_bloods"], firsts)

        perf_score = (
            weight_ka * norm_ka +
            weight_deaths * norm_deaths +
            weight_first * norm_first
        )

        # Adjust score based on ELO — lower ELO → boosted score
        elo_diff = avg_elo - p["elo"]  # invert: low elo → positive diff
        elo_scale = 1.0 + elo_weight * (elo_diff / 400)
        adjusted_score = perf_score * elo_scale

        raw_scores.append(adjusted_score)

    # Normalize
    total_score = sum(raw_scores)
    perf_weights = [s / total_score for s in raw_scores] if total_score else [1 / n] * n

    # Distribute ELO gain
    perf_deltas = [round(remaining_delta * w, 6) for w in perf_weights]
    deltas = [round(p + floor_per_player, 2) for p in perf_deltas]

    # Fix rounding drift
    diff = round(team_delta - sum(deltas), 2)
    if diff != 0:
        idx = max(range(n), key=lambda i: deltas[i])
        deltas[idx] += diff

    return deltas


def compute_team_elo_losses_with_floor(team_stats, team_delta,
                                       weight_ka=0.6, weight_deaths=0.1, weight_first=0.2,
                                       floor_percent=0.1, elo_weight=0.5):
    """
    For a LOSS: distributes a fixed total ELO loss (team_delta < 0)
    Penalizes poor performers more, scaled further by their ELO relative to team.
    """
    assert team_delta < 0, "team_delta must be negative for losses"
    n = len(team_stats)
    floor_total = team_delta * floor_percent
    remaining_delta = team_delta - floor_total
    floor_per_player = round(floor_total / n, 6)

    # Stat vectors
    kas = [p["kills"] * 1.5 + p["assists"] * 0.5 for p in team_stats]
    deaths = [p["deaths"] for p in team_stats]
    firsts = [p["first_bloods"] for p in team_stats]
    elos = [p["elo"] for p in team_stats]
    avg_elo = sum(elos) / n

    # Compute inverse performance and adjust by relative elo
    raw_scores = []
    for i, p in enumerate(team_stats):
        norm_ka = normalize_stat(kas[i], kas)
        norm_deaths = 1 - normalize_stat(p["deaths"], deaths)
        norm_first = normalize_stat(p["first_bloods"], firsts)

        perf_score = (
            weight_ka * norm_ka +
            weight_deaths * norm_deaths +
            weight_first * norm_first
        )

        inverse_perf = 1.0 - perf_score

        # Adjust score by relative ELO (higher ELO = larger weight)
        elo_diff = p["elo"] - avg_elo
        elo_scale = 1.0 + elo_weight * (elo_diff / 400)  # 400 is standard ELO scaling
        adjusted_score = inverse_perf * elo_scale

        raw_scores.append(adjusted_score)

    # Normalize scores
    total_score = sum(raw_scores)
    perf_weights = [s / total_score for s in raw_scores] if total_score else [1 / n] * n

    # Allocate negative ELO
    perf_deltas = [round(remaining_delta * w, 6) for w in perf_weights]
    deltas = [round(p + floor_per_player, 2) for p in perf_deltas]

    # Fix rounding drift
    diff = round(team_delta - sum(deltas), 2)
    if diff != 0:
        idx = min(range(n), key=lambda i: deltas[i])
        deltas[idx] += diff

    return deltas

def change_rank(user, new_rank):
    today = datetime.now().date()

    today_entry = (Player
                   .select()
                   .where((Player.player == user) &
                          (fn.DATE(Player.updated_at) == today))
                   .first())
    
    if today_entry:
        today_entry.rank = new_rank
        today_entry.updated_at = datetime.now()
        today_entry.save()
        print(f"Updated today's ELO for {user} to {new_rank}")
    else:
        Player.create(player=user, rank=new_rank, updated_at=datetime.now())
        print(f"Created new ELO entry for {user} at {new_rank} (new day)")



def main():
    init_db()
    files = detect_files()
    for file in files:
        print(file)
        rounds_won, rounds_lost, result = detect_gamescore(file)
        rounds_won, rounds_lost = int(rounds_won), int(rounds_lost)
        print(f"Rounds Won: {rounds_won}, Rounds Lost: {rounds_lost}, Result: {result}")
    
        green_team, red_team = detect_scorelines(file, green_win = True if result.lower() == "victorl" else False)

        print("WINNING TEAM SCORELINES")
        for i in range(len(green_team)):
            print(f"Username: {green_team[i][0]}, Combat Score: {green_team[i][1]}, Kills: {green_team[i][2]}, Deaths: {green_team[i][3]}, Assists: {green_team[i][4]}, First Bloods: {green_team[i][5]}")

        print("LOSING TEAM SCORELINES")
        for i in range(len(red_team)):
            print(f"Username: {red_team[i][0]}, Combat Score: {red_team[i][1]}, Kills: {red_team[i][2]}, Deaths: {red_team[i][3]}, Assists: {red_team[i][4]}, First Bloods: {red_team[i][5]}")

        add_to_match_history(green_team, file, "win")
        add_to_match_history(red_team, file, "loss")

        winningTeamUsers = [player[0] for player in green_team]
        losingTeamUsers = [player[0] for player in red_team]
        player_hashmap = {}
        all_current_elos(player_hashmap, winningTeamUsers + losingTeamUsers)

        winningTeamElo = team_elo(winningTeamUsers, player_hashmap)
        losingTeamElo = team_elo(losingTeamUsers, player_hashmap)
        print(f"Winning Team ELO: {winningTeamElo}")
        print(f"Losing Team ELO: {losingTeamElo}")

        elo_A, elo_B, wA, wB = skew_relative_team_elo(winningTeamElo, losingTeamElo, alpha=1.5)

        print("Skewed Team A ELO:", elo_A)
        print("Skewed Team B ELO:", elo_B)
        print("Team A Weights:", wA)
        print("Team B Weights:", wB)

        winning_percentage = expected_score(elo_A, elo_B)
        print(f"Winning Percentage: {winning_percentage}")

        even_points = points_for_even(rounds_won, rounds_lost)
        print(f"Points for even: {even_points}")

        winning_team_elo_change = elo_change(elo_A, elo_B, 1, k=even_points)
        losing_team_elo_change = elo_change(elo_B, elo_A, 0, k=even_points)
        print(f"Winning Team ELO Change: {winning_team_elo_change}")
        print(f"Losing Team ELO Change: {losing_team_elo_change}")

        winning_team_list = [{"player": player[0], "kills": int(player[2]), "assists": int(player[4]), "deaths": int(player[3]), 
                              "first_bloods": int(player[5]), "elo": player_hashmap[player[0]][0]} for player in green_team]
        losing_team_list = [{"player": player[0], "kills": int(player[2]), "assists": int(player[4]), "deaths": int(player[3]), 
                             "first_bloods": int(player[5]), "elo": player_hashmap[player[0]][0]} for player in red_team]

        winning_team_elo_deltas = compute_team_elo_deltas_with_floor(winning_team_list, winning_team_elo_change * 5, floor_percent=0.5)
        losing_team_elo_deltas = compute_team_elo_losses_with_floor(losing_team_list, losing_team_elo_change * 5, floor_percent=0.5)
        print("Winning Team ELO Deltas:", winning_team_elo_deltas)
        print("Losing Team ELO Deltas:", losing_team_elo_deltas)

        for i, player in enumerate(green_team):
            player_name = player[0].replace(" ", "")
            new_elo = player_hashmap[player_name][0] + winning_team_elo_deltas[i]
            change_rank(player_name, new_elo)

        for i, player in enumerate(red_team):
            player_name = player[0].replace(" ", "")
            new_elo = player_hashmap[player_name][0] + losing_team_elo_deltas[i]
            change_rank(player_name, new_elo)

        # Save game to database
        game = Game.create(filename=file, processed_at=datetime.now())
        game.save()
        print(f"Processed and saved game: {file}")

        new_elos = {}
        for player in winningTeamUsers + losingTeamUsers:
            player_entry = (Player
                            .select()
                            .where(Player.player == player)
                            .order_by(Player.updated_at.desc())
                            .first())
            if player_entry:
                new_elos[player] = [player_entry.rank, player_entry.updated_at]
                print(f"{player} - {player_entry.rank} - {player_entry.updated_at}")
            else:
                print(f"Player {player} does not exist in the database.")


if __name__ == "__main__":
    main()
