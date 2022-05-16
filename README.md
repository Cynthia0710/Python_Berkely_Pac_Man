# Python_Cynthia0710-Python_Berkely-Pac-Man

## Project 1

### Part 1

Iterative Deepening Search algorithm: 

`python pacman.py -l mediumMaze -p SearchAgent -a fn=ids`

### Part 2

Weighted A* algorithm:

`python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=wastar,heuristic=manhattanHeuristic`

### Part 3

`python pacman.py -l capsuleSearch -p CapsuleSearchAgent -a fn=wastar,prob=CapsuleSearchProblem,heuristic=foodHeuristic`

## Project 2

Contest: Pacman Capture the Flag (Contest with baseline)

Winning: A game ends when one team returns all but two of the opponents' dots. Games are also limited to 1200 agent moves (300 moves per each of the four agents). If this move limit is reached, whichever team has returned the most food wins. If the score is zero (i.e., tied) this is recorded as a tie game.

`python capture.py -r baselineTeam -b baselineTeam/myTeam -l RANDOM13`

## Requirement

if there is an error " ImportError: No module named 'Tkinter' " and you're using python 3.9 on Mac, you can simply install tkinter using brew:

`brew install python-tk@3.9`
