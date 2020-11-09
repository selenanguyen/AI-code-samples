import sys
import copy
import numpy as np
from queue import PriorityQueue

PUZZLE_WIDTH = 4
BLANK = 0  # Integer comparison tends to be faster than string comparison
BETTER = True
START_KEY = -1000 # Must be lower than any other key values

# Read a NumberPuzzle from stdin; space-delimited, blank is "-"
def read_puzzle():
    new_puzzle = NumberPuzzle()
    row = 0
    for line in sys.stdin:
        tokens = line.split()
        for i in range(PUZZLE_WIDTH):
            if tokens[i] == '-':
                new_puzzle.tiles[row][i] = BLANK
                new_puzzle.blank_r = row
                new_puzzle.blank_c = i
            else:
                try:
                    new_puzzle.tiles[row][i] = int(tokens[i])
                except ValueError:
                    sys.exit("Found unexpected non-integer for tile value")
        row += 1
    return new_puzzle

# Class containing an int array for tiles
# blank_c and blank_r for column and row of the blank
# (so we don't have to hunt for it)
class NumberPuzzle:
    # This is a constructor in Python - just return zeros for everything
    # and fill in the tile array later
    def __init__(self):
        self.tiles = np.zeros((PUZZLE_WIDTH,PUZZLE_WIDTH))
        self.blank_r = 0
        self.blank_c = 0
        # This next field is for our convenience when generating a solution
        # -- remember which puzzle was the move before
        self.parent = None
        self.dist_from_start = 0
        self.key = 0

    # This is the Python equivalent of Java's toString()
    def __str__(self):
        out = ""
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if j > 0:
                    out += " "
                if self.tiles[i][j] == BLANK:
                    out += "-"
                else:
                    out += str(int(self.tiles[i][j]))
            out += "\n"
        return out

    # In A* search, we generally want to copy instead of destructively alter,
    # since we're not backtracking so much as jumping around the search tree.
    # Also, if A and B are numpy arrays, "A = B" only passes a reference to B.
    # "deepcopy" copies out the data (recursively if need be, a fact possibly
    # useful for other assignments).
        # We'll also use this to tell the child we're its parent
    def copy(self):
        child = NumberPuzzle()
        child.tiles = np.copy(self.tiles)
        child.blank_r = self.blank_r
        child.blank_c = self.blank_c
        child.dist_from_start = self.dist_from_start
        child.parent = self
        return child

    # Overrides == for this object so that we can compare by tile arrangement
    # instead of reference.  This is going to be pretty common, so we'll skip
    # a type check on "other" for a modest speed increase
    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)

    # Hash function necessary for inclusion in a set -- unique "name"
    # for this object -- we'll just hash the bytes of the tile array
    def __hash__(self):
        return hash(bytes(self.tiles))

    # Override less-than so that we can put these in a priority queue
    # with no problem.  We don't want to recompute the heuristic here, 
    # though -- that would be too slow to do it every time we need to
    # reorganize the priority queue 

    def __lt__(self, obj):
        return (self.key < obj.key) or (self.key == obj.key and self.dist_from_start > obj.dist_from_start)

    # Move from the row, column coordinates given into the blank.
    # Also very common, so we will also skip checks for legality to improve
    # speed.
    def move(self, tile_row, tile_column):
        self.tiles[self.blank_r][self.blank_c] = self.tiles[tile_row][tile_column]
        self.tiles[tile_row][tile_column] = BLANK
        self.blank_r = tile_row
        self.blank_c = tile_column
        self.dist_from_start += 1

    # Return a list of NumberPuzzle states that could result from one
    # move on the present board.  Use this to keep the order in which
    # moves are evaluated the same as our solution, thus matching the
    # HackerRank solution as well.  (Also notice we're still in the
    # methods of NumberPuzzle, hence the lack of arguments.)
    def legal_moves(self):
        legal = []
        if (self.blank_r > 0):
            down_result = self.copy()
            down_result.move(self.blank_r-1, self.blank_c)
            legal.append(down_result)
        if (self.blank_c > 0):
            right_result = self.copy()
            right_result.move(self.blank_r, self.blank_c-1)
            legal.append(right_result)
        if (self.blank_r < PUZZLE_WIDTH - 1):
            up_result = self.copy()
            up_result.move(self.blank_r+1, self.blank_c)
            legal.append(up_result)
        if (self.blank_c < PUZZLE_WIDTH - 1):
            left_result = self.copy()
            left_result.move(self.blank_r, self.blank_c+1)
            legal.append(left_result)
        return legal
        
    # solve: return a list of puzzle states from this state to solved
    # better_h flag determines whether to use the better heuristic
    def solve(self, better_h):
        closed_list = set([])
        queue = PriorityQueue()
        self.key = START_KEY # set to -1000
        queue.put(self)
        while queue._qsize() != 0:
            state = queue._get()
            # if state is solved, return the list of states as the solution
            if state.solved():
                solution = [state]
                # climb back up the tree to find the path
                parent = state.parent
                while parent is not None:
                    solution.insert(0, parent)
                    parent = parent.parent
                return solution
            # if we have explored this state already, continue
            if state.__hash__() in closed_list:
                continue
            # otherwise, add this state to the closed list and add its neighbors to the queue
            closed_list.add(state.__hash__())
            for neighbor in state.legal_moves():
                # neighbor's key = cost to get there + heuristic
                neighbor.key = neighbor.heuristic(better_h) + neighbor.dist_from_start
                queue.put(neighbor)
                # set the neighbor's parent to the current state
                neighbor.parent = state
        return None
    
    # solved(): return True iff all tiles in order and blank in bottom right
    def solved(self):
        should_be = 1
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if self.tiles[i][j] != should_be:
                    return False
                else:
                    should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return True

    # Toggle which heuristic is used
    def heuristic(self, better_h):
        if better_h:
            return self.manhattan_heuristic()
        return self.tile_mismatch_heuristic()

    # Count tiles out of place.
    def tile_mismatch_heuristic(self):
        mismatch_count = 0
        should_be = 1
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if self.tiles[i][j] != should_be and self.tiles[i][j] != BLANK:
                    mismatch_count = mismatch_count + 1
                should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return mismatch_count

    # Returns total manhattan (city block) distance needed for all tiles.  
    def manhattan_heuristic(self):
        total_manhattan = 0
        should_be = 1
        # tiles we have found that are mismatched
        found = {}
        # tiles we know are mismatched, but not yet found
        not_found = {}
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                actual = self.tiles[i][j]
                # if the tile is mismatched...
                if actual != should_be:
                    # if we've passed the correct location of the tile placed here, remove it from not_found
                    # and add its manhattan distance
                    if actual in not_found:
                        total_manhattan = total_manhattan + abs(i - not_found[actual][0]) + abs(j - not_found[actual][1])
                        del not_found[actual]
                    # otherwise, add it to the found dict if the tile is not the blank tile
                    elif actual != BLANK:
                        found[actual] = (i, j)
                    # if we have already found the tile that should be placed here, remove it from found
                    # and add its manhattan distance
                    if should_be in found:
                        total_manhattan = total_manhattan + abs(i - found[should_be][0]) + abs(j - found[should_be][1])
                        del found[should_be]
                    # otherwise, if not the blank tile, add it to not_found
                    elif should_be != BLANK:
                        not_found[should_be] = (i, j)
                should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return total_manhattan

# Print every puzzle in the list
def print_steps(path):
    if path is None:
        print("No path found")
    else:
        for state in path:
            print(state)

# main
my_puzzle = read_puzzle()
solution_steps = my_puzzle.solve(BETTER)
print_steps(solution_steps)
