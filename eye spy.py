import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 800
HEIGHT = 600

TOTAL_LEVELS = 7

DWELL_TIME = 0.8

MIN_X = 170
MAX_X = 630

MIN_Y = 140
MAX_Y = 470

OBJECT_RADIUS = 34


# ============================================================
# CAMERA
# ============================================================

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 60)

if not cam.isOpened():
    raise RuntimeError("Camera could not be opened.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# GAZE
# ============================================================

smooth_x = WIDTH // 2
smooth_y = HEIGHT // 2

alpha = 0.30


# ============================================================
# LEVEL THEMES
# ============================================================

themes = [

    {
        "name": "FACES",
        "normal": "😊",
        "different": "😎"
    },

    {
        "name": "FRUITS",
        "normal": "🍎",
        "different": "🍊"
    },

    {
        "name": "SPACE",
        "normal": "★",
        "different": "◆"
    },

    {
        "name": "SEA",
        "normal": "🐟",
        "different": "🐙"
    },

    {
        "name": "SHAPES",
        "normal": "●",
        "different": "◆"
    },

    {
        "name": "ALIENS",
        "normal": "👽",
        "different": "🤖"
    },

    {
        "name": "SYMBOLS",
        "normal": "+",
        "different": "×"
    }
]


# ============================================================
# GAME VARIABLES
# ============================================================

level = 1

score = 0

game_over = False

objects = []

different_index = 0

hover_index = None

hover_start = None

level_start_time = time.time()

message = ""

message_until = 0


# ============================================================
# CREATE LEVEL
# ============================================================

def create_level():

    global objects
    global different_index
    global hover_index
    global hover_start
    global level_start_time

    objects = []

    # More objects as levels increase
    number_of_objects = min(
        5 + level,
        12
    )

    # Create objects
    while len(objects) < number_of_objects:

        x = random.randint(
            MIN_X,
            MAX_X
        )

        y = random.randint(
            MIN_Y,
            MAX_Y
        )

        valid = True

        for obj in objects:

            distance = math.sqrt(
                (x - obj["x"]) ** 2 +
                (y - obj["y"]) ** 2
            )

            if distance < 85:

                valid = False
                break

        if valid:

            objects.append({
                "x": x,
                "y": y
            })

    different_index = random.randint(
        0,
        len(objects) - 1
    )

    hover_index = None

    hover_start = None

    level_start_time = time.time()


# ============================================================
# RESET
# ============================================================

def reset_game():

    global level
    global score
    global game_over
    global message
    global message_until

    level = 1

    score = 0

    game_over = False

    message = ""

    message_until = 0

    create_level()


# ============================================================
# GAZE TRACKING
# ============================================================

def get_gaze(frame):

    global smooth_x
    global smooth_y

    small = cv2.resize(
        frame,
        (320, 240)
    )

    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    )

    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:

        return None

    landmarks = (
        result
        .multi_face_landmarks[0]
        .landmark
    )

    # MediaPipe iris landmarks
    left_iris = landmarks[468]
    right_iris = landmarks[473]

    iris_x = (
        left_iris.x +
        right_iris.x
    ) / 2

    iris_y = (
        left_iris.y +
        right_iris.y
    ) / 2

    # Keep gaze movement inside a comfortable area
    mapped_x = np.interp(
        iris_x,
        [0.35, 0.65],
        [MIN_X, MAX_X]
    )

    mapped_y = np.interp(
        iris_y,
        [0.35, 0.65],
        [MIN_Y, MAX_Y]
    )

    # Smooth movement
    smooth_x += (
        mapped_x -
        smooth_x
    ) * alpha

    smooth_y += (
        mapped_y -
        smooth_y
    ) * alpha

    smooth_x = np.clip(
        smooth_x,
        0,
        WIDTH - 1
    )

    smooth_y = np.clip(
        smooth_y,
        0,
        HEIGHT - 1
    )

    return (
        int(smooth_x),
        int(smooth_y)
    )


# ============================================================
# FIND OBJECT UNDER GAZE
# ============================================================

def find_object(gaze_x, gaze_y):

    closest = None

    closest_distance = 999999

    for i, obj in enumerate(objects):

        distance = math.sqrt(
            (gaze_x - obj["x"]) ** 2 +
            (gaze_y - obj["y"]) ** 2
        )

        if distance < OBJECT_RADIUS + 18:

            if distance < closest_distance:

                closest_distance = distance

                closest = i

    return closest


# ============================================================
# DRAW OBJECT
# ============================================================

def draw_object(
    frame,
    x,
    y,
    symbol,
    theme_name
):

    # Different rendering for shapes/symbols
    if theme_name in ["SHAPES", "SPACE", "SYMBOLS"]:

        if symbol == "●":

            cv2.circle(
                frame,
                (x, y),
                OBJECT_RADIUS,
                (80, 190, 255),
                -1
            )

        elif symbol == "◆":

            points = np.array([
                [x, y - OBJECT_RADIUS],
                [x + OBJECT_RADIUS, y],
                [x, y + OBJECT_RADIUS],
                [x - OBJECT_RADIUS, y]
            ])

            cv2.fillPoly(
                frame,
                [points],
                (255, 100, 200)
            )

        elif symbol == "★":

            # Star
            pts = []

            for i in range(10):

                angle = (
                    -math.pi / 2
                    + i * math.pi / 5
                )

                radius = (
                    OBJECT_RADIUS
                    if i % 2 == 0
                    else OBJECT_RADIUS * 0.45
                )

                px = int(
                    x + math.cos(angle) * radius
                )

                py = int(
                    y + math.sin(angle) * radius
                )

                pts.append(
                    [px, py]
                )

            cv2.fillPoly(
                frame,
                [np.array(pts)],
                (100, 220, 255)
            )

        else:

            cv2.putText(
                frame,
                symbol,
                (x - 25, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (255, 220, 100),
                4
            )

    else:

        # Emoji-like symbols using Unicode
        # fallback drawing

        if theme_name == "FACES":

            if symbol == "😊":

                # Face
                cv2.circle(
                    frame,
                    (x, y),
                    OBJECT_RADIUS,
                    (0, 210, 255),
                    -1
                )

                # Eyes
                cv2.circle(
                    frame,
                    (x - 11, y - 8),
                    4,
                    (0, 0, 0),
                    -1
                )

                cv2.circle(
                    frame,
                    (x + 11, y - 8),
                    4,
                    (0, 0, 0),
                    -1
                )

                # Smile
                cv2.ellipse(
                    frame,
                    (x, y + 4),
                    (13, 8),
                    0,
                    0,
                    180,
                    (0, 0, 0),
                    3
                )

            else:

                # Sunglasses face

                cv2.circle(
                    frame,
                    (x, y),
                    OBJECT_RADIUS,
                    (0, 210, 255),
                    -1
                )

                cv2.rectangle(
                    frame,
                    (x - 20, y - 12),
                    (x - 2, y + 2),
                    (20, 20, 20),
                    -1
                )

                cv2.rectangle(
                    frame,
                    (x + 2, y - 12),
                    (x + 20, y + 2),
                    (20, 20, 20),
                    -1
                )

                cv2.line(
                    frame,
                    (x - 2, y - 5),
                    (x + 2, y - 5),
                    (20, 20, 20),
                    3
                )

                cv2.ellipse(
                    frame,
                    (x, y + 6),
                    (13, 8),
                    0,
                    0,
                    180,
                    (0, 0, 0),
                    3
                )

        elif theme_name == "FRUITS":

            if symbol == "🍎":

                # Apple
                cv2.circle(
                    frame,
                    (x - 12, y),
                    23,
                    (30, 50, 220),
                    -1
                )

                cv2.circle(
                    frame,
                    (x + 12, y),
                    23,
                    (30, 50, 220),
                    -1
                )

                cv2.line(
                    frame,
                    (x, y - 20),
                    (x + 5, y - 35),
                    (30, 150, 50),
                    5
                )

            else:

                # Orange
                cv2.circle(
                    frame,
                    (x, y),
                    OBJECT_RADIUS,
                    (0, 150, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    (x - 8, y - 10),
                    3,
                    (255, 200, 100),
                    -1
                )

        elif theme_name == "SEA":

            if symbol == "🐟":

                # Fish
                pts = np.array([
                    [x - 28, y],
                    [x, y - 22],
                    [x + 28, y],
                    [x, y + 22]
                ])

                cv2.fillPoly(
                    frame,
                    [pts],
                    (80, 180, 255)
                )

                cv2.circle(
                    frame,
                    (x + 12, y - 5),
                    4,
                    (0, 0, 0),
                    -1
                )

            else:

                # Octopus
                cv2.circle(
                    frame,
                    (x, y - 8),
                    25,
                    (180, 80, 220),
                    -1
                )

                for dx in [-18, -6, 6, 18]:

                    cv2.line(
                        frame,
                        (x + dx, y + 10),
                        (x + dx, y + 30),
                        (180, 80, 220),
                        7
                    )

        elif theme_name == "ALIENS":

            if symbol == "👽":

                # Alien
                cv2.ellipse(
                    frame,
                    (x, y),
                    (30, 38),
                    0,
                    0,
                    360,
                    (80, 220, 120),
                    -1
                )

                cv2.ellipse(
                    frame,
                    (x - 10, y - 5),
                    (7, 12),
                    0,
                    0,
                    360,
                    (20, 20, 30),
                    -1
                )

                cv2.ellipse(
                    frame,
                    (x + 10, y - 5),
                    (7, 12),
                    0,
                    0,
                    360,
                    (20, 20, 30),
                    -1
                )

            else:

                # Robot
                cv2.rectangle(
                    frame,
                    (x - 28, y - 30),
                    (x + 28, y + 30),
                    (120, 150, 180),
                    -1
                )

                cv2.circle(
                    frame,
                    (x - 10, y - 5),
                    6,
                    (0, 255, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    (x + 10, y - 5),
                    6,
                    (0, 255, 255),
                    -1
                )


# ============================================================
# START
# ============================================================

reset_game()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cam.read()

        if not ret:
            break

        frame = cv2.flip(
            frame,
            1
        )

        gaze = get_gaze(frame)

        game = np.zeros(
            (
                HEIGHT,
                WIDTH,
                3
            ),
            dtype=np.uint8
        )

        # Background
        game[:] = (
            10,
            15,
            28
        )

        theme = themes[level - 1]

        theme_name = theme["name"]


        # ====================================================
        # HEADER
        # ====================================================

        cv2.putText(
            game,
            "EYE SPY",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )

        cv2.putText(
            game,
            f"LEVEL {level}",
            (350, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            game,
            f"Score: {score}",
            (650, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # THEME
        # ====================================================

        cv2.putText(
            game,
            theme_name,
            (350, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 190, 220),
            1
        )


        # ====================================================
        # INSTRUCTION
        # ====================================================

        if not game_over:

            cv2.putText(
                game,
                "Find the DIFFERENT object",
                (275, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 220, 220),
                2
            )


        # ====================================================
        # DRAW OBJECTS
        # ====================================================

        for i, obj in enumerate(objects):

            if i == different_index:

                symbol = theme["different"]

            else:

                symbol = theme["normal"]

            draw_object(
                game,
                obj["x"],
                obj["y"],
                symbol,
                theme_name
            )


        # ====================================================
        # GAZE
        # ====================================================

        if gaze is not None and not game_over:

            gaze_x, gaze_y = gaze

            current = find_object(
                gaze_x,
                gaze_y
            )


            if current is not None:

                target = objects[current]

                # Highlight
                cv2.circle(
                    game,
                    (
                        target["x"],
                        target["y"]
                    ),
                    OBJECT_RADIUS + 9,
                    (0, 255, 255),
                    3
                )


                # New object
                if hover_index != current:

                    hover_index = current

                    hover_start = time.time()


                else:

                    elapsed = (
                        time.time()
                        - hover_start
                    )

                    progress = min(
                        elapsed / DWELL_TIME,
                        1.0
                    )

                    # Progress ring
                    cv2.ellipse(
                        game,
                        (
                            target["x"],
                            target["y"]
                        ),
                        (
                            OBJECT_RADIUS + 15,
                            OBJECT_RADIUS + 15
                        ),
                        -90,
                        0,
                        int(360 * progress),
                        (255, 0, 255),
                        5
                    )


                    # Selection complete
                    if elapsed >= DWELL_TIME:

                        if current == different_index:

                            score += level * 10

                            message = (
                                f"CORRECT!  "
                                f"LEVEL {level} COMPLETE"
                            )

                            message_until = (
                                time.time() + 1.2
                            )

                            if level >= TOTAL_LEVELS:

                                game_over = True

                            else:

                                level += 1

                                create_level()


                        else:

                            score = max(
                                0,
                                score - 5
                            )

                            message = (
                                "WRONG! TRY AGAIN"
                            )

                            message_until = (
                                time.time() + 1.0
                            )

                            hover_index = None

                            hover_start = None


            else:

                hover_index = None

                hover_start = None


            # Gaze indicator
            cv2.circle(
                game,
                (
                    gaze_x,
                    gaze_y
                ),
                6,
                (0, 255, 255),
                -1
            )

            cv2.circle(
                game,
                (
                    gaze_x,
                    gaze_y
                ),
                13,
                (0, 180, 255),
                2
            )


        else:

            if not game_over:

                cv2.putText(
                    game,
                    "Face not detected",
                    (310, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )


        # ====================================================
        # MESSAGE
        # ====================================================

        if (
            message != ""
            and
            time.time() < message_until
        ):

            cv2.putText(
                game,
                message,
                (220, 555),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


        # ====================================================
        # GAME COMPLETE
        # ====================================================

        if game_over:

            overlay = game.copy()

            cv2.rectangle(
                overlay,
                (150, 190),
                (650, 420),
                (5, 5, 12),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.9,
                game,
                0.1,
                0,
                game
            )

            cv2.putText(
                game,
                "YOU COMPLETED ALL LEVELS!",
                (205, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )

            cv2.putText(
                game,
                f"FINAL SCORE: {score}",
                (285, 305),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                game,
                "Excellent Eye Control!",
                (285, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (150, 255, 180),
                2
            )

            cv2.putText(
                game,
                "R = Restart    Q = Quit",
                (275, 395),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )


        # ====================================================
        # FOOTER
        # ====================================================

        if not game_over:

            cv2.putText(
                game,
                "Look at an object and hold your gaze",
                (270, 580),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (160, 160, 160),
                2
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Eye Spy - Multi Level",
            game
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        if key == ord("r"):

            reset_game()


finally:

    cam.release()

    face_mesh.close()

    cv2.destroyAllWindows()