# Reward code from the Unsloth Gemma 4 (E2B) Sudoku RL notebook (cells [10, 18, 20, 33, 35, 37, 39]), unchanged except
# for the helper import from cell 25.
from unsloth import check_python_modules, create_locked_down_function


from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import copy

def _is_valid_placement(board: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if placing num at (row, col) is valid."""
    # Check row
    if num in board[row]:
        return False

    # Check column
    if num in [board[r][col] for r in range(9)]:
        return False

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False

    return True

def _solve_sudoku(board: List[List[int]]) -> bool:
    """Solve sudoku using backtracking (for puzzle generation)."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if _is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if _solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def _generate_complete_board(rng: random.Random) -> List[List[int]]:
    """Generate a complete valid Sudoku board."""
    board = [[0 for _ in range(9)] for _ in range(9)]

    # Fill diagonal 3x3 boxes first (they don't affect each other)
    for box in range(3):
        nums = list(range(1, 10))
        rng.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]

    # Solve the rest
    _solve_sudoku(board)
    return board

@dataclass
class SudokuGame:
    difficulty: int = 40  # Number of cells to remove (20 = easy, 40 = medium, 50 = hard)
    seed: Optional[int] = None
    _rng: random.Random = field(init = False, repr = False)
    _board: List[List[int]] = field(init = False, repr = False)
    _solution: List[List[int]] = field(init = False, repr = False)
    _initial_board: List[List[int]] = field(init = False, repr = False)
    _moves: int = field(default = 0, init = False, repr = False)
    _state: str = field(default = "ongoing", init = False, repr = False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

        # Generate complete board
        complete_board = _generate_complete_board(self._rng)
        self._solution = copy.deepcopy(complete_board)

        # Remove cells to create puzzle
        self._board = copy.deepcopy(complete_board)
        cells = [(r, c) for r in range(9) for c in range(9)]
        self._rng.shuffle(cells)

        for r, c in cells[:self.difficulty]:
            self._board[r][c] = 0

        self._initial_board = copy.deepcopy(self._board)
        self._update_state()

    def board(self) -> List[List[int]]:
        """Return current board state."""
        return [row[:] for row in self._board]

    def initial_board(self) -> List[List[int]]:
        """Return initial puzzle state."""
        return [row[:] for row in self._initial_board]

    def state(self) -> str:
        """Return game state: 'ongoing', 'success', or 'failed'."""
        return self._state

    def moves(self) -> int:
        """Return number of moves made."""
        return self._moves

    def place_number(self, row: int, col: int, num: int) -> bool:
        """Place a number on the board. Returns True if valid move."""
        # Validate input
        if not (0 <= row < 9 and 0 <= col < 9):
            self._state = "failed"
            return False

        if not (1 <= num <= 9):
            self._state = "failed"
            return False

        # Can't modify initial cells
        if self._initial_board[row][col] != 0:
            self._state = "failed"
            return False
        if self._board[row][col] != 0:
            self._state = "failed"
            return False
        # Check if placement is valid
        if not _is_valid_placement(self._board, row, col, num):
            self._state = "failed"
            return False

        # Place number
        self._board[row][col] = num
        self._moves += 1
        self._update_state()
        return True

    def _update_state(self) -> None:
        """Update game state based on current board."""
        # Check if puzzle is complete
        if all(self._board[r][c] != 0 for r in range(9) for c in range(9)):
            # Verify solution is correct
            if self._board == self._solution:
                self._state = "success"
            else:
                self._state = "failed"
        else:
            self._state = "ongoing"

    def pretty(self, colors: bool = True) -> str:
        """Pretty print the Sudoku board."""
        RESET = "\x1b[0m"
        INITIAL = "\x1b[38;5;45m"   # Cyan for initial numbers
        PLACED = "\x1b[38;5;226m"    # Yellow for placed numbers
        EMPTY = "\x1b[38;5;239m"     # Gray for empty cells

        lines = []
        lines.append("┌───────┬───────┬───────┐")

        for row in range(9):
            row_str = "│ "
            for col in range(9):
                num = self._board[row][col]

                if colors:
                    if num == 0:
                        row_str += f"{EMPTY}.{RESET}"
                    elif self._initial_board[row][col] != 0:
                        row_str += f"{INITIAL}{num}{RESET}"
                    else:
                        row_str += f"{PLACED}{num}{RESET}"
                else:
                    row_str += str(num) if num != 0 else "."

                if col % 3 == 2:
                    row_str += " │ "
                else:
                    row_str += " "

            lines.append(row_str.rstrip())

            if row == 8:
                lines.append("└───────┴───────┴───────┘")
            elif row % 3 == 2:
                lines.append("├───────┼───────┼───────┤")

        return "\n".join(lines)


from typing import Callable
from unsloth import execute_with_time_limit

def _execute_strategy(strategy: Callable, game: SudokuGame):
    """Execute a strategy function on a Sudoku game."""
    assert callable(strategy)

    max_moves = 100
    valid_moves = 0  # Track successful moves

    while game.state() == "ongoing" and valid_moves < max_moves:
        try:
            board = game.board()
            initial = game.initial_board()
            result = strategy(board, initial)

            # Validate result format
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                # Invalid format = immediate fail, but return valid moves made
                return valid_moves, "failed"

            row, col, num = result

            # Validate types
            if not all(isinstance(x, int) for x in [row, col, num]):
                return valid_moves, "failed"

            # Try to place number
            success = game.place_number(row, col, num)

            if success:
                valid_moves += 1  # Count this valid move
            else:
                # Invalid move = game fails, but return valid_moves made so far
                return valid_moves, "failed"

        except Exception:
            return valid_moves, "failed"

    if valid_moves >= max_moves and game.state() == "ongoing":
        return valid_moves, "failed"

    return valid_moves, game.state()


@execute_with_time_limit(10)
def execute_strategy(strategy: Callable, game: SudokuGame):
    """Execute strategy with 10 second time limit."""
    return _execute_strategy(strategy, game)


def extract_function(text):
    """Extract Python function from markdown code blocks."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board, initial):"):
            return fx
    return None


def function_works(completions, **kwargs):
    """Reward for generating valid executable Python code."""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)

        if function is not None:
            ok, info = check_python_modules(function)

        if function is None or "error" in info:
            score = -2.0  # Invalid function
        else:
            try:
                new_strategy = create_locked_down_function(function)
                score = 1.0  # Valid function
            except:
                score = -1.0  # Function has errors

        scores.append(score)
    return scores


def no_cheating(completions, **kwargs):
    """Penalize use of external imports."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)

        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)  # Heavy penalty for cheating
        else:
            scores.append(-1.0)  # Failed to create function

    return scores


import numpy as np

global PRINTER
PRINTER = 0

def strategy_succeeds(completions, **kwargs):
    """Reward valid moves even if strategy eventually fails."""
    global PRINTER
    scores = []

    seed = np.random.randint(10000)
    difficulty = 40
    for completion in completions:
        printed = False
        response = completion[0]["content"]
        function = extract_function(response)

        if PRINTER % 5 == 0:
            printed = True
            print("\n" + "=" * 60)
            print(function)
            print("=" * 60)
        PRINTER += 1

        if function is not None:
            ok, info = check_python_modules(function)

        if function is None or "error" in info:
            scores.append(0)
            continue

        try:
            new_strategy = create_locked_down_function(function)
        except:
            scores.append(0)
            continue

        try:
            game = SudokuGame(difficulty = difficulty, seed = seed)
            valid_moves, game_state = execute_strategy(new_strategy, game)
            if valid_moves == difficulty:
                game_state = "success"

            print(f"\n Valid moves: {valid_moves}, Final state: {game_state}")

            if not printed:
                print("Strategy:")
                print(function[:200] + "..." if len(function) > 200 else function)

            print("\nFinal board:")
            print(game.pretty())

            if game_state == "success":
                scores.append(30.0)  # Solved the puzzle!
            elif valid_moves > 0:
                # Reward based on valid moves made before failure
                # Each valid move is worth 0.2 points
                reward = valid_moves * 0.2
                scores.append(reward)
            else:
                scores.append(-2.0)  # Failed immediately with no valid moves

        except TimeoutError:
            print("Timeout")
            scores.append(-1.0)
        except Exception as e:
            print(f"Exception: {str(e)[:100]}")
            scores.append(-3.0)

    return scores
