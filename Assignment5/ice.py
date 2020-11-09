# "MDPs on Ice - Assignment 5"
# Ported from Java

import random
import numpy as np
import copy
import sys

MIN_UTILITY = -1000
GOLD_REWARD = 100.0
PIT_REWARD = -150.0
DISCOUNT_FACTOR = 0.5
EXPLORE_PROB = 0.2 # for Q-learning
LEARNING_RATE = 0.1
ITERATIONS = 10000
MAX_MOVES = 1000
ACTIONS = 4
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
MOVES = ['U','R','D','L']

# Fixed random number generator seed for result reproducibility --
# don't use a random number generator besides this to match sol
random.seed(5100)


# Problem class:  represents the physical space, transition probabilities, reward locations,
# and approach to use (MDP or Q) - in short, the info in the text file
class Problem:
    # Fields:
    # approach - string, "MDP" or "Q"
    # move_probs - list of doubles, probability of going 1,2,3 spaces
    # map - list of list of strings: "-" (safe, empty space), "G" (gold), "P" (pit)

    # Format looks like
    # MDP    [approach to be used]
    # 0.7 0.2 0.1   [probability of going 1, 2, 3 spaces]
    # - - - - - - P - - - -   [space-delimited map rows]
    # - - G - - - - - P - -   [G is gold, P is pit]
    #
    # You can assume the maps are rectangular, although this isn't enforced
    # by this constructor.

    # __init__ consumes stdin; don't call it after stdin is consumed or outside that context
    def __init__(self):
        self.approach = input('Reading mode...')
        print(self.approach)
        probs_string = input("Reading transition probabilities...\n")
        self.move_probs = [float(s) for s in probs_string.split()]
        self.map = []
        for line in sys.stdin:
            self.map.append(line.split())

    def solve(self, iterations):            
        if self.approach == "MDP":
            return mdp_solve(self, iterations)
        elif self.approach == "Q":
            return q_solve(self, iterations)
        return None
        
# Policy: Abstraction on the best action to perform in each state - just a 2D string list-of-lists
class Policy:
    def __init__(self, problem): # problem is a Problem
        # Signal 'no policy' by just displaying the map there
        self.best_actions = copy.deepcopy(problem.map)

    def __str__(self):
        return '\n'.join([' '.join(row) for row in self.best_actions])

# roll_steps:  helper for try_policy and q_solve -- "rolls the dice" for the ice and returns
# the new location (r,c), taking map bounds into account
# note that move is expecting a string, not an integer constant
def roll_steps(move_probs, row, col, move, rows, cols):
    displacement = 1
    total_prob = 0
    move_sample = random.random()
    for p, prob in enumerate(problem.move_probs):
        total_prob += prob
        if move_sample <= total_prob:
            displacement = p+1
            break
    # Handle "slipping" into edge of map
    new_row = row
    new_col = col
    if not isinstance(move,str):
        print("Warning: roll_steps wants str for move, got a different type")
    if move == "U":
        new_row -= displacement
        if new_row < 0:
            new_row = 0
    elif move == "R":
        new_col += displacement
        if new_col >= cols:
            new_col = cols-1
    elif move == "D":
        new_row += displacement
        if new_row >= rows:
            new_row = rows-1
    elif move == "L":
        new_col -= displacement
        if new_col < 0:
            new_col = 0
    return new_row, new_col


# try_policy:  returns avg utility per move of the policy, as measured by "iterations"
# random drops of an agent onto empty spaces, running until gold, pit, or time limit 
# MAX_MOVES is reached
def try_policy(policy, problem, iterations):
    total_utility = 0
    total_moves = 0
    for i in range(iterations):
        # Resample until we have an empty starting square
        while True:
            row = random.randrange(0,len(problem.map))
            col = random.randrange(0,len(problem.map[0]))
            if problem.map[row][col] == "-":
                break
        for moves in range(MAX_MOVES):
            total_moves += 1
            policy_rec = policy.best_actions[row][col]
            # Take the move - roll to see how far we go, bump into map edges as necessary
            row, col = roll_steps(problem.move_probs, row, col, policy_rec, len(problem.map), len(problem.map[0]))
            if problem.map[row][col] == "G":
                total_utility += GOLD_REWARD
                break
            if problem.map[row][col] == "P":
                total_utility -= PIT_REWARD
                break
    return total_utility / total_moves

# get_validated_coord: helper function that returns the actual board location reached, given the 
# potentially out-of-bounds location reached after slipping on the ice
# (used this function in mdp_solve before discovering the roll_steps function)
def get_validated_coord(coord, num_rows, num_cols):
    row = coord[0]
    col = coord[1]
    if row < 0:
        row = 0
    if col < 0:
        col = 0
    if row >= num_rows:
        row = num_rows - 1
    if col >= num_cols:
        col = num_cols - 1
    return (row, col)

# mdp_solve:  use [iterations] iterations of the Bellman equations over the whole map in [problem]
# and return the policy of what action to take in each square
def mdp_solve(problem, iterations):
    policy = Policy(problem)
    num_rows = len(problem.map)
    num_cols = len(problem.map[1])
    prev_utilities = []
    # initialize the utilities table
    for row in range(num_rows):
        prev_utilities.append([])
        for col in range(num_cols):
            if problem.map[row][col] == '-':
                prev_utilities[row].append(0)
            elif problem.map[row][col] == 'P':
                prev_utilities[row].append(PIT_REWARD)
            elif problem.map[row][col] == 'G':
                prev_utilities[row].append(GOLD_REWARD)
    # begin iterations
    for i in range(iterations):
        # initialize a new utilities table at every iteration
        utilities = []
        for row in range(num_rows):
            utilities.append([])
            for col in range(num_cols):
                if problem.map[row][col] != "-":
                    utilities[row].append(prev_utilities[row][col])
                    continue
                # find the max utility reached using possible moves
                max_utility = MIN_UTILITY
                for move in MOVES:
                    new_coords = []
                    for x in range(len(problem.move_probs)):
                        new_coords.append([row, col])
                    if move == 'U':
                        for x in range(len(new_coords)):
                            new_coords[x][0] -= (x + 1)
                    elif move == "R":
                        for x in range(len(new_coords)):
                            new_coords[x][1] += (x + 1)
                    elif move == "D":
                        for x in range(len(new_coords)):
                            new_coords[x][0] += (x + 1)
                    elif move == "L":
                        for x in range(len(new_coords)):
                            new_coords[x][1] -= (x + 1)
                    for x in range(len(new_coords)):
                        new_coords[x] = list(get_validated_coord(tuple(new_coords[x]), num_rows, num_cols))
                    utility = 0
                    for x in range(len(problem.move_probs)):
                        utility += problem.move_probs[x] * prev_utilities[new_coords[x][0]][new_coords[x][1]]
                    utility = utility * DISCOUNT_FACTOR
                    # update policy based on max utility
                    if utility > max_utility:
                        max_utility = utility
                        policy.best_actions[row][col] = move
                # update current utilities board with max utility
                utilities[row].append(max_utility) 
        prev_utilities = utilities
    return policy

# q_solve:  use [iterations] iterations of the Bellman equations over the whole map in [problem]
# and return the policy of what action to take in each square
def q_solve(problem, iterations):
    policy = Policy(problem)
    num_rows = len(problem.map)
    num_cols = len(problem.map[1])
    # initialize the q-value board
    q_values = []
    for row in range(num_rows):
        q_values.append([])
        for col in range(num_cols):
            if problem.map[row][col] == '-':
                q_values[row].append({
                    'U': 0,
                    'R': 0,
                    'D': 0,
                    'L': 0
                })
            elif problem.map[row][col] == 'P':
                q_values[row].append({
                    'U': PIT_REWARD,
                    'R': PIT_REWARD,
                    'D': PIT_REWARD,
                    'L': PIT_REWARD
                })
            elif problem.map[row][col] == 'G':
                q_values[row].append({
                    'U': GOLD_REWARD,
                    'R': GOLD_REWARD,
                    'D': GOLD_REWARD,
                    'L': GOLD_REWARD
                })
    # begin iterations
    for i in range(iterations):
        # choose a random starting state
        starting_s = (random.randint(0, num_rows - 1), random.randint(0, num_cols - 1))
        # if the starting state is a pit or gold, end this iteration
        if problem.map[starting_s[0]][starting_s[1]] != '-':
            continue
        # keep updating states reached along the selected path, until we reach a pit or gold or the max number
        # of moves is reached
        num_moves = 0
        while problem.map[starting_s[0]][starting_s[1]] == '-' and num_moves < MAX_MOVES:
            # set the next action to the max action from this state
            action = 'U'
            row = starting_s[0]
            col = starting_s[1]
            max_q = q_values[row][col]['U']
            for move in MOVES:
                if q_values[row][col][move] > max_q:
                    action = move
                    max_q = q_values[row][col][move]
            # or explore in a random direction with probability EXPLORE_PROB
            if random.random() < EXPLORE_PROB:
                action = MOVES[random.randint(0, 3)]
            # get new location after slipping
            next_row, next_col = roll_steps(problem.move_probs, row, col, action, num_rows, num_cols)
            next_s = (next_row, next_col)
            max_q_next = MIN_UTILITY
            # use the q-value of the next state to calculate the updated q-value of the current state
            for move in MOVES:
                if q_values[next_row][next_col][move] > max_q_next:
                    max_q_next = q_values[next_row][next_col][move]
            q_values[row][col][action] += LEARNING_RATE * (DISCOUNT_FACTOR * max_q_next - q_values[row][col][action])
            max_q = MIN_UTILITY
            # update the policy based on the max q-value for this state
            for move in MOVES:
                if q_values[row][col][move] > max_q:
                    max_q = q_values[row][col][move]
                    policy.best_actions[row][col] = move
            starting_s = (next_row, next_col)
            num_moves += 1
    return policy

# Main:  read the problem from stdin, print the policy and the utility over a test run
if __name__ == "__main__":
    problem = Problem()
    policy = problem.solve(ITERATIONS)
    print(policy)
    print("Calculating average utility...")
    print("Average utility per move: {utility:.2f}".format(utility = try_policy(policy, problem,ITERATIONS)))
        
