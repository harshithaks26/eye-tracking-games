import subprocess
import sys


print()
print("================================")
print("       GAZEPLAY")
print("  EYE-TRACKING GAME PLATFORM")
print("================================")
print()

print("1. Gaze Maze")
print("2. Gaze Target Challenge")
print("3. Exit")

print()

choice = input(
    "Choose a game: "
).strip()


if choice == "1":

    subprocess.run(
        [
            sys.executable,
            "gaze_maze.py"
        ]
    )

elif choice == "2":

    subprocess.run(
        [
            sys.executable,
            "gaze_target.py"
        ]
    )

else:

    print("Exiting...")