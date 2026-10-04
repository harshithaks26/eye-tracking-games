
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ============================================================
# EYE-CONTROLLED TIC-TAC-TOE
# Features:
# - MediaPipe iris tracking
# - Gaze -> 3x3 board mapping
# - Smooth gaze tracking
# - Adaptive dwell time
# - AI opponent using Minimax
# - Gaze heatmap / game statistics
# ============================================================

# ---------------- CAMERA ----------------
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 60)

if not cam.isOpened():
    print("ERROR: Camera could not be opened.")
    print("On macOS, allow camera access for VS Code/Terminal in:")
    print("System Settings -> Privacy & Security -> Camera")
    raise SystemExit

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- BOARD ----------------
board = [["", "", ""],
         ["", "", ""],
         ["", "", ""]]

PLAYER = "X"
AI = "O"
current_player = PLAYER

winner = None
game_over = False

# ---------------- BOARD SETTINGS ----------------
WIDTH = 600
HEIGHT = 600
CELL = 200

# ---------------- GAZE SETTINGS ----------------
smooth_x = WIDTH / 2
smooth_y = HEIGHT / 2

alpha = 0.35

# Recent gaze points are used to estimate stability.
gaze_history = deque(maxlen=12)

# Base dwell time.
BASE_DWELL = 1.5
MIN_DWELL = 0.9
MAX_DWELL = 2.0

hover_cell = None
hover_start = None

# ---------------- ANALYTICS ----------------
heatmap = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]

selection_times = []
total_moves = 0

# ---------------- AI ----------------
AI_DELAY = 0.7
ai_move_time = None


# ============================================================
# DRAWING
# ============================================================

def draw_background(frame):
    frame[:] = (8, 8, 8)

    for r in range(3):
        for c in range(3):
            x1 = c * CELL + 5
            y1 = r * CELL + 5
            shade = (25, 25, 25) if (r + c) % 2 == 0 else (15, 15, 15)
            cv2.rectangle(
                frame,
                (x1, y1),
                (x1 + CELL - 10, y1 + CELL - 10),
                shade,
                -1
            )


def draw_grid(frame):
    for x in [CELL, CELL * 2]:
        cv2.line(frame, (x, 0), (x, HEIGHT), (220, 220, 220), 3)

    for y in [CELL, CELL * 2]:
        cv2.line(frame, (0, y), (WIDTH, y), (220, 220, 220), 3)


def draw_moves(frame):
    for r in range(3):
        for c in range(3):
            value = board[r][c]

            if value == "":
                continue

            x = c * CELL + 62
            y = r * CELL + 140

            if value == "X":
                color = (0, 180, 255)
            else:
                color = (255, 220, 80)

            cv2.putText(
                frame,
                value,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.5,
                color,
                7
            )


def highlight_cell(frame, row, col):
    x1 = col * CELL + 7
    y1 = row * CELL + 7
    x2 = (col + 1) * CELL - 7
    y2 = (row + 1) * CELL - 7

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 230, 255),
        5
    )


def draw_heatmap(frame):
    max_value = max(max(row) for row in heatmap)

    if max_value == 0:
        return

    for r in range(3):
        for c in range(3):
            value = heatmap[r][c]

            if value == 0:
                continue

            # Transparent red overlay based on gaze frequency.
            intensity = int(40 + 140 * (value / max_value))
            overlay = frame.copy()

            x1 = c * CELL + 20
            y1 = r * CELL + 20
            x2 = (c + 1) * CELL - 20
            y2 = (r + 1) * CELL - 20

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, intensity),
                -1
            )

            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_status(frame, text, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# ============================================================
# GAME LOGIC
# ============================================================

def check_winner(state):
    for r in range(3):
        if state[r][0] != "" and state[r][0] == state[r][1] == state[r][2]:
            return state[r][0]

    for c in range(3):
        if state[0][c] != "" and state[0][c] == state[1][c] == state[2][c]:
            return state[0][c]

    if state[0][0] != "" and state[0][0] == state[1][1] == state[2][2]:
        return state[0][0]

    if state[0][2] != "" and state[0][2] == state[1][1] == state[2][0]:
        return state[0][2]

    return None


def board_full(state):
    return all(cell != "" for row in state for cell in row)


def empty_cells(state):
    cells = []

    for r in range(3):
        for c in range(3):
            if state[r][c] == "":
                cells.append((r, c))

    return cells


def minimax(state, maximizing):
    result = check_winner(state)

    if result == AI:
        return 1

    if result == PLAYER:
        return -1

    if board_full(state):
        return 0

    if maximizing:
        best_score = -999

        for r, c in empty_cells(state):
            state[r][c] = AI
            score = minimax(state, False)
            state[r][c] = ""
            best_score = max(best_score, score)

        return best_score

    best_score = 999

    for r, c in empty_cells(state):
        state[r][c] = PLAYER
        score = minimax(state, True)
        state[r][c] = ""
        best_score = min(best_score, score)

    return best_score


def best_ai_move():
    best_score = -999
    best_move = None

    for r, c in empty_cells(board):
        board[r][c] = AI
        score = minimax(board, False)
        board[r][c] = ""

        if score > best_score:
            best_score = score
            best_move = (r, c)

    return best_move


def reset_game():
    global board, current_player, winner, game_over
    global hover_cell, hover_start, total_moves
    global heatmap, selection_times, ai_move_time
    global gaze_history

    board = [["", "", ""],
             ["", "", ""],
             ["", "", ""]]

    current_player = PLAYER
    winner = None
    game_over = False

    hover_cell = None
    hover_start = None

    total_moves = 0
    selection_times = []

    heatmap = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]

    ai_move_time = None
    gaze_history.clear()


def make_player_move(row, col, elapsed):
    global current_player, winner, game_over
    global total_moves, hover_start

    if board[row][col] != "":
        return

    board[row][col] = PLAYER
    total_moves += 1
    selection_times.append(elapsed)

    # Check after player's move.
    winner = check_winner(board)

    if winner or board_full(board):
        game_over = True
        return

    current_player = AI


def make_ai_move():
    global current_player, winner, game_over
    global total_moves

    move = best_ai_move()

    if move is None:
        game_over = True
        return

    row, col = move
    board[row][col] = AI
    total_moves += 1

    winner = check_winner(board)

    if winner or board_full(board):
        game_over = True
        return

    current_player = PLAYER


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, camera_frame = cam.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    camera_frame = cv2.flip(camera_frame, 1)

    small = cv2.resize(camera_frame, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    game = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    draw_background(game)
    draw_grid(game)
    draw_heatmap(game)
    draw_moves(game)

    # --------------------------------------------------------
    # PLAYER TURN: EYE TRACKING
    # --------------------------------------------------------
    if current_player == PLAYER and not game_over:

        if result.multi_face_landmarks:

            landmarks = result.multi_face_landmarks[0].landmark

            # MediaPipe iris landmarks.
            left_iris = landmarks[468]
            right_iris = landmarks[473]

            iris_x = (left_iris.x + right_iris.x) / 2
            iris_y = (left_iris.y + right_iris.y) / 2

            # Map gaze to board.
            mapped_x = np.interp(
                iris_x,
                [0.35, 0.65],
                [0, WIDTH - 1]
            )

            mapped_y = np.interp(
                iris_y,
                [0.35, 0.65],
                [0, HEIGHT - 1]
            )

            # Smooth gaze.
            smooth_x += (mapped_x - smooth_x) * alpha
            smooth_y += (mapped_y - smooth_y) * alpha

            smooth_x = float(np.clip(smooth_x, 0, WIDTH - 1))
            smooth_y = float(np.clip(smooth_y, 0, HEIGHT - 1))

            # Track recent gaze positions.
            gaze_history.append((smooth_x, smooth_y))

            # Gaze stability.
            if len(gaze_history) >= 5:
                points = np.array(gaze_history)
                spread = float(np.mean(np.std(points, axis=0)))
            else:
                spread = 20.0

            # Lower spread = more stable = faster selection.
            stability_factor = np.clip(spread / 15.0, 0, 1)

            dwell_time = MIN_DWELL + (
                MAX_DWELL - MIN_DWELL
            ) * stability_factor

            row = int(smooth_y // CELL)
            col = int(smooth_x // CELL)

            if 0 <= row < 3 and 0 <= col < 3:

                heatmap[row][col] += 1

                highlight_cell(game, row, col)

                current_hover = (row, col)

                # Start/restart dwell when gaze enters another cell.
                if hover_cell != current_hover:
                    hover_cell = current_hover
                    hover_start = time.time()

                elif board[row][col] == "":
                    elapsed = time.time() - hover_start

                    # Dwell progress.
                    progress = min(elapsed / dwell_time, 1.0)

                    bar_x = col * CELL + 15
                    bar_y = row * CELL + 175

                    cv2.rectangle(
                        game,
                        (bar_x, bar_y),
                        (bar_x + 170, bar_y + 10),
                        (50, 50, 50),
                        -1
                    )

                    cv2.rectangle(
                        game,
                        (bar_x, bar_y),
                        (bar_x + int(170 * progress), bar_y + 10),
                        (0, 220, 255),
                        -1
                    )

                    cv2.putText(
                        game,
                        f"Dwell: {elapsed:.1f}s",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        game,
                        f"Stability: {max(0, 100 - int(spread * 5))}%",
                        (20, 92),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 255, 180),
                        2
                    )

                    if elapsed >= dwell_time:

                        make_player_move(row, col, elapsed)

                        hover_start = None
                        hover_cell = None
                        gaze_history.clear()

            # Gaze cursor.
            cv2.circle(
                game,
                (int(smooth_x), int(smooth_y)),
                12,
                (0, 255, 255),
                -1
            )

            cv2.circle(
                game,
                (int(smooth_x), int(smooth_y)),
                4,
                (255, 255, 255),
                -1
            )

        else:
            draw_status(
                game,
                "Face not detected - look at the camera",
                (0, 0, 255)
            )

    # --------------------------------------------------------
    # AI TURN
    # --------------------------------------------------------
    elif current_player == AI and not game_over:

        draw_status(
            game,
            "Computer is thinking...",
            (255, 220, 80)
        )

        if ai_move_time is None:
            ai_move_time = time.time()

        if time.time() - ai_move_time >= AI_DELAY:

            make_ai_move()

            ai_move_time = None
            hover_start = None
            hover_cell = None

    # --------------------------------------------------------
    # GAME OVER
    # --------------------------------------------------------
    if game_over:

        overlay = game.copy()

        cv2.rectangle(
            overlay,
            (55, 190),
            (545, 410),
            (15, 15, 15),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.85,
            game,
            0.15,
            0,
            game
        )

        if winner == PLAYER:
            result_text = "YOU WIN!"
            result_color = (0, 255, 255)

        elif winner == AI:
            result_text = "AI WINS!"
            result_color = (100, 180, 255)

        else:
            result_text = "DRAW!"
            result_color = (200, 200, 200)

        cv2.putText(
            game,
            result_text,
            (145, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            result_color,
            4
        )

        avg_time = (
            sum(selection_times) / len(selection_times)
            if selection_times else 0
        )

        max_heat = max(max(row) for row in heatmap)

        most_viewed = "None"

        if max_heat > 0:
            for r in range(3):
                for c in range(3):
                    if heatmap[r][c] == max_heat:
                        most_viewed = f"Row {r + 1}, Column {c + 1}"

        cv2.putText(
            game,
            f"Moves: {total_moves}",
            (130, 315),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            game,
            f"Avg gaze time: {avg_time:.2f}s",
            (130, 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            game,
            f"Most viewed: {most_viewed}",
            (130, 375),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------------
    # TOP / BOTTOM UI
    # --------------------------------------------------------
    if not game_over:
        turn_text = "YOUR TURN (X)" if current_player == PLAYER else "AI TURN (O)"

        cv2.putText(
            game,
            turn_text,
            (WIDTH - 235, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    cv2.putText(
        game,
        "R = Restart   Q = Quit",
        (180, 585),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        2
    )

    cv2.imshow("Eye-Controlled Tic-Tac-Toe", game)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r"):
        reset_game()


# ---------------- CLEANUP ----------------
cam.release()
face_mesh.close()
cv2.destroyAllWindows()
