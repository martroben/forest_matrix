import os
import random
import time
from typing import Callable


##################
# Helper Classes #
##################

class CharacterManipulation:
    """
    Class to aggregate character-related functions.
    """

    # ANSI escape commands
    ESCAPE: str = "\x1b"                    # ANSI escape code denoting that a command follows
    SET_COLOUR_COMMAND: str = "38;5;"       # ANSI code denoting character colour setting
    SETTING_START: str = "["                # ANSI code denoting the start of command
    SETTING_END: str = "m"                  # ANSI code denoting the end of command
    HOME: str = "H"                         # ANSI code to return to position 1, 1 in terminal
    BLANK_CHARACTER: str = " "

    OBFUSCATION_REGISTER: dict[str, str] = {
        "a": ["@", "4", "Д"],
        "b": ["ß", "Ъ"],
        "c": ["©"],
        "d": ["Ð", "đ"],
        "e": ["€", "3", "Є"],
        "i": ["1", "Ї"],
        "l": ["1", "|", "£"],
        "o": ["0", "ø", "°", "Ѳ"],
        "n": ["и"],
        "r": ["®", "я"],
        "s": ["$", "5", "§"],
        "t": ["+", "†", "7"],
        "u": ["µ", "Ц"],
        "x": ["×"], 
    }

    @staticmethod
    def get_coloured_character(character: str, colour256: int) -> str:
        """
        Attaches ANSI escape command to add colour to input character.
        """
        return f"{CharacterManipulation.ESCAPE}{CharacterManipulation.SETTING_START}{CharacterManipulation.SET_COLOUR_COMMAND}{colour256}{CharacterManipulation.SETTING_END}{character}"
    
    @staticmethod
    def return_to_top() -> str:
        """
        Returns a character that takes the cursor back to position 1, 1 in terminal.
        """
        return f"{CharacterManipulation.ESCAPE}{CharacterManipulation.SETTING_START}{CharacterManipulation.HOME}"
    
    @staticmethod
    def get_obfuscated_text(text: str, obfuscation_probability: float) -> str:
        """
        Obfustactes random letters in input text with visually similar characters.
        """
        obfuscated_letters = []
        for letter in text:
            if (letter not in CharacterManipulation.OBFUSCATION_REGISTER.keys()) or (random.random() > obfuscation_probability):
                obfuscated_letters += [letter]
                continue
            obfuscated_letters += [random.choice(CharacterManipulation.OBFUSCATION_REGISTER[letter])]
        return "".join(obfuscated_letters)


class GradualChange:
    """
    Class that gives gradual probability transitions for smooth changes in animation.
    """
    def __init__(self, start_probability: float, end_probability: float, n_steps: int, exponent: float = 3) -> None:
        self.start_probability = start_probability
        self.end_probability = end_probability
        self.n_steps = n_steps
        self.exponent = exponent
    
    def function(self, x: float, y1: float, y2: float) -> float:
        """
        Core polynomial function.
        """
        return (x / self.n_steps)**self.exponent * (y2 - y1) + y1

    def get_accelerating_probability(self, i_step: int) -> float:
        """
        Accelerating probability function: small changes in the beginning, big in the end.
        """
        if i_step >= self.n_steps:
            return self.end_probability
        return self.function(i_step, self.start_probability, self.end_probability)
    
    def get_decelerating_probability(self, i_step: int) -> float:
        """
        Decelerating probability function: big changes in the beginning, small in the end.
        """
        if i_step >= self.n_steps:
            return self.end_probability
        return self.function(self.n_steps - i_step, self.end_probability, self.start_probability)


class AsciiImage:
    """
    Class for loading and transforming an Ascii image.
    """
    BLANK_CHARACTER: str = " "
    NEWLINE_CHARACTER: str = "\n"

    def __init__(self, ascii_text: str) -> None:
        self.text: str = ascii_text

    def get_binary(self) -> list[list[bool]]:
        """
        Return a matrix (list or rows) where character locations are indicated by True values.
        """
        binary_image = []
        for row in self.text.split(self.NEWLINE_CHARACTER):
            binary_row = [character != self.BLANK_CHARACTER for character in row]
            binary_image += [binary_row]
        return binary_image

    def get_scaled_matrix(self, n_rows: int, n_columns: int) -> list[list[bool]]:
        """
        Position binary image in the middle of a matrix of given dimensions. I.e. pad it with False cells.
        """
        binary = self.get_binary()
        n_rows_image = len(binary)
        n_columns_image = len(binary[0])

        if n_rows_image > n_rows or n_columns_image > n_columns:
            # raise ValueError("Scaling dimensions are smaller than the input image. Function can only scale the number of rows and columns up, not down.")
            # Return just an empty result instead of raising an error
            return [[]]

        n_pad_rows_top = (n_rows - n_rows_image) // 2
        n_pad_rows_bottom = n_rows - n_pad_rows_top - n_rows_image
        n_pad_columns_left = (n_columns - n_columns_image) // 2
        n_pad_columns_right = n_columns - n_pad_columns_left - n_columns_image

        scaled_image_midpart = [n_pad_columns_left * [False] + row + n_pad_columns_right * [False] for row in binary]
        scaled_image = n_pad_rows_top * [n_columns * [False]] + scaled_image_midpart + n_pad_rows_bottom * [n_columns * [False]]

        return scaled_image


##################
# Matrix classes #
##################

class Drop:
    """
    Object representing a raindrop in matrix rain.
    """
    def __init__(self, length: int) -> None:
        self.length: int = length
        self.step: int = 1           # Parameter for possible extension: could be used to make drops that move several cells in a frame
    
    def get_colour(self, position_in_drop: int, bright_colours: int, lit_colours: list[int], fading_colours: list[int]) -> int:
        """
        Takes the position of a cell within the drop and returns the colour that the cell should take in current frame.
        """
        if position_in_drop == 0:
            return random.choice(bright_colours)
        # sequence of all colors
        colour_sequence = lit_colours + fading_colours
        # If the brightness bias is close to 1, the drop colour is biased towards brighter colours. I.e. the colours towards the beginning og the sequence
        brightness_bias: float = 0.7    # Values from 0 to 1
        i_colour = int(((len(colour_sequence) - 1) * position_in_drop / self.length - brightness_bias) // 1 + 1)
        return colour_sequence[i_colour]
    
    def get_next_position(self, position_in_drop: int) -> int:
        """
        Takes the current position of a cell in the drop and returns what position this cell will be in the next frame.
        """
        next_position = position_in_drop + self.step
        if next_position < self.length:
            return next_position
        return None


class Cell:
    """
    Object representing a cell (a single character space) in the matrix.
    """
    # Colour codes in Ascii 256 colour system
    BRIGHT_COLOURS: list[int] = [231]           # whites
    LIT_COLOURS: list[int] = [48, 41, 35, 29]   # greens
    DIM_COLOURS: list[int] = [29, 22]           # dark greens
    FADING_COLOURS: list[int] = [238]           # grays
    INIVISIBLE_COLOUR: int = -1                 # black (color code 0 doesn't look good on screen, so we return a blank character instead)

    def __init__(self, character: str) -> None:
        self.character: str = character
        self.default_colour: int = random.choice(self.LIT_COLOURS)

        self.is_lit: bool = False               # Variable determining whether the cell should be visible in current frame

        self.position_in_drop: int = 0          # Cell position in a Drop, starting from drop head. 0-based indexing
        self.drop: Drop = None

        self.override_colour: int = None        # Used for applying colour changing glitches to cell
        self.override_character: str = None     # Used for applying message to cell

        self.is_ascii_image: bool = False       # Cell is part of a 2d ascii image
        self.is_message: bool = False           # Cell is part of a vertical text "message"

    def __str__(self) -> str:
        """
        Returns the cell character in correct colour if it is visible (i.e. "lit") or blank character if it's not visible.
        """
        if self.is_lit:
            active_character = self.get_active_character()
            active_colour = self.get_active_colour()
            return CharacterManipulation.get_coloured_character(active_character, active_colour)
        return CharacterManipulation.BLANK_CHARACTER

    def get_active_colour(self):
        """
        Returns the colour of the character based on whether it's currently part of a rain drop or an ongoing glitch.
        """
        if self.override_colour:
            return self.override_colour
        if self.drop:
            drop_colour = self.drop.get_colour(self.position_in_drop, self.BRIGHT_COLOURS, self.LIT_COLOURS, self.FADING_COLOURS)
            return drop_colour
        return self.default_colour

    def get_active_character(self):
        """
        Returns the default character or override character of the cell - or blank character if cell colour is set to invisible.
        """
        # Black doesn't look good on screen, so we return a blank character instead
        if self.get_active_colour() == self.INIVISIBLE_COLOUR:
            return CharacterManipulation.BLANK_CHARACTER
        return self.override_character or self.character

    def set_drop_head(self, drop_length: int) -> None:
        """
        Sets the cell as the first cell of an incoming drop.
        """
        self.position_in_drop = 0
        self.drop = Drop(drop_length)
        self.is_lit = True

    def move_drop(self, image_active: bool) -> None:
        """
        Re-assigns the cell position in a rain drop for the next frame.
        """
        if not self.drop:
            # Cell is not part of an active drop
            return
        if next_position := self.drop.get_next_position(self.position_in_drop):
            # If next_position is not None, the cell is part of drop body / tail
            self.position_in_drop = next_position
            return
        # Else, drop has passed the cell and it's set back to inactive stage
        self.drop = None
        # Set cell as not lit, unless it's part of an active ascii image
        self.is_lit = image_active and self.is_ascii_image


class Glitch:
    """
    Object representing single-cell glitches occuring in the matrix.
    """
    def __init__(self, cell: Cell) -> None:
        self.cell: Cell = cell
        self.action_queue: list[Callable] = []      # A list of cell transformations to be carried out as part of the glitch

        # Don't glitch messages
        if cell.is_message:
            return

        self.action_queue += random.choice([self.burnout(), self.flicker_colour(), self.flicker_character()])
        self.action_queue += [self.clear]
        # Reverse, so it can be applied by list.pop()
        self.action_queue = list(reversed(self.action_queue))
    
    def do_action(self) -> None:
        """
        Performs tne next transformation step on a cell and then removes it from the queue.
        """
        action = self.action_queue.pop()
        action()

    # Single transformations, i.e. actions on the glitched cell. Building blocks for action sequences
    def flash(self) -> None:
        self.cell.override_colour = random.choice(self.cell.BRIGHT_COLOURS)

    def invisible(self) -> None:
        self.cell.override_colour = self.cell.INIVISIBLE_COLOUR
    
    def dim(self) -> None:
        self.cell.override_colour = random.choice(self.cell.DIM_COLOURS)
    
    def change_character(self) -> None:
        self.cell.character = random.choice(Matrix.AVAILABLE_CHARACTERS)

    def sleep(self) -> None:
        return

    def clear(self) -> list:
        self.cell.override_colour = None

    # Action sequences
    def flicker_colour(self) -> list:
        """
        Change cell colour between random dim colours repeatedly.
        """
        return random.randint(5, 20) * ([self.dim] + random.randint(5, 10) * [self.sleep])
    
    def flicker_character(self) -> list:
        """
        Change cell character repeatedly.
        """
        return random.randint(5, 10) * ([self.change_character] + 10 * [self.sleep])
    
    def burnout(self) -> list:
        """
        Apply a bright colour for a few frames and then make cell invisible for a period.
        """
        # Go bright and then dark for a period, before reappearing again
        return [self.flash] + random.randint(1, 4) * [self.sleep] + random.randint(5, 20) * [self.invisible] + [self.change_character]


class Message:
    """
    Object representing the hidden messages appearing vertically in the matrix characters.
    """
    def __init__(self, cells: list[tuple[Cell, str]]) -> None:
        self.cells: list[tuple[Cell, str]] = cells
        # Set random order of cells to apply the reveal / hide actions
        self.action_queue: list[tuple[Cell, str]] = random.sample(self.cells, k=len(self.cells))

        self.sleep: int = 0                             # Variable for delaying the message action steps
        self.action: tuple[Cell, str] = None            # Ongoing action
        self.deleted: bool = False                      # Indicator to carry out the message deletion sequence
    
    @staticmethod
    def set_override_character(cell: Cell, character: str) -> None:
        """
        Set cell override character to reveal or hide the message.
        """
        # If character is None (i.e. deletion sequence), remove message indicator from cell
        cell.is_message = character is not None
        cell.override_character = character
    
    def do_action(self) -> None:
        """
        Perform a step in the message revealing / hiding sequence.
        """
        # While the ongoing action is in sleep phase, count down sleep steps
        if self.sleep:
            self.sleep -= 1
            return
        
        # If there is no ongoing action and no queue, do nothing
        if not (self.action_queue or self.action):
            return
        
        # If there is no ongoing action, start the next one from queue
        if not self.action:
            self.action = self.action_queue.pop()
            self.set_override_character(self.action[0], CharacterManipulation.BLANK_CHARACTER)
            # Apply random blank period for every action
            self.sleep = random.randint(0, 5)
        
        # If the ongoing action has completed the sleep phase, set the message character and clear the action variable
        self.set_override_character(*self.action)
        self.action = None

    def delete(self) -> None:
        """
        Set a sequence for deleting the message.
        """
        # Avoid restarting ongoing delete
        if not self.deleted:
            # Set random deletion order for cells
            self.action_queue = random.choices([(cell, None) for cell, _ in self.cells], k=len(self.cells))
            self.deleted = True


class Matrix:
    """
    Object representing the onscreen matrix (rows and cells).
    """
    MIN_DROP_LENGTH: int = 4
    MAX_DROP_LENGTH: int = 25
    DROP_PROBABLITY: float = 0.01                           # Drop probablity per column per step
    GLITCH_PROBABILITY: float = 0.0002                      # Glitch probability per cell per step
    N_CONCURRENT_MESSAGES: int = 40                         # Number of messages active at any time
    MESSAGE_REPLACE_PROBABLITY: float = 0.001               # Probablity that an existing message is deleted and another one spawned per frame
    MESSAGE_OBFUSCATION_PROBABILITY: float = 0.25           # Probability of letter obfuscation per letter in message

    # Forestry related symbols: ϙ Ѧ ⍋ ⍦ ☙ ⚐ ⚘ ⚲ ⚶ ✿ ❀ ❦ ❧ ⲯ ⸙ 🙖 🜎
    CHARACTER_CODE_POINTS: list[int] = [985, 1126, 9035, 9062, 9753, 9872, 9880, 9906, 9910, 10047, 10048, 10086, 10087, 11439, 11801, 128598, 128782]
    AVAILABLE_CHARACTERS: list[int] = [chr(x) for x in CHARACTER_CODE_POINTS]

    def __init__(self, n_rows: int, n_columns: int) -> None:
        self.n_rows: int = n_rows
        self.n_columns: int = n_columns

        self.rows: list[list[Cell]] = []                    # Variable representing a list of rows consisting of cells
        self.glitches: list[Glitch] = []                    # List of active glitches
        self.messages: list[tuple[Message, int]] = []       # Message object and the index of the column it's applied to (allows to control that every column only has one message)
        self.active_drop_probability: float = self.DROP_PROBABLITY      # Duplicated drop probability variable is set, because it changes when "stopping" the rain
        self.rain_active: bool = True                       # Variable to indicate if the rain stop sequence should be started

        # Populate the matrix
        for _ in range(self.n_rows):
            row = [Cell(character) for character in random.choices(self.AVAILABLE_CHARACTERS, k=self.n_columns)]
            self.rows.append(row)

    def __str__(self) -> str:
        """
        Returns the string that is printed on screen.
        """
        return "".join("".join(str(cell) for cell in row) for row in self.rows)
    
    def set_ascii_image(self, ascii_image: AsciiImage) -> None:
        """
        Sets an ascii image that can be revealed during the animation.
        """
        ascii_image_matrix: list[list[bool]] = ascii_image.get_scaled_matrix(self.n_rows, self.n_columns)
        for i_row, image_row in enumerate(ascii_image_matrix):
            for i_column, is_ascii_image in enumerate(image_row):
                self.rows[i_row][i_column].is_ascii_image = is_ascii_image
        
        # Make a register of ascii image top edge cells to be used when "washing" away the image
        self.image_top_cells = []
        for i_column in range(len(self.rows[0])):
            for i_row, row in enumerate(self.rows):
                cell = row[i_column]
                if cell.is_ascii_image:
                    self.image_top_cells += [cell]
                    continue

        self.ascii_image_active = False

    def move_drops(self) -> None:
        """
        Cycles through all cells in the matrix and updates their positions in drops to advance frame.
        """
        # Iterate through rows from second to last
        # Iteration starts from the bottom row, because next frame of a cell depends on the state of the cell above
        for i_row_above, row in reversed(list(enumerate(self.rows[1:]))):
            for i_column, current_cell in enumerate(row):
                # Advance frame of each cell
                current_cell.move_drop(image_active = self.ascii_image_active)
                # If the cell above is drop head, set the current cell as drop head for next frame
                cell_above = self.rows[i_row_above][i_column]
                if cell_above.drop and cell_above.position_in_drop == 0:
                    current_cell.set_drop_head(drop_length=cell_above.drop.length)
        
        # Iterate through first row separately, because there is no row above
        for first_row_cell in self.rows[0]:
            first_row_cell.move_drop(image_active = self.ascii_image_active)

    def spawn_drops(self) -> None:
        """
        Spawn new drops in the first row with currently active drop probability.
        """
        for cell in self.rows[0]:
            if random.random() > self.active_drop_probability:
                continue

            drop_length = random.randint(self.MIN_DROP_LENGTH, self.MAX_DROP_LENGTH)
            cell.set_drop_head(drop_length)

    def spawn_ascii_image_washing_drops(self) -> None:
        """
        Spawn new drops in the top boundary of an Ascii image if the image is no longer active (to "wash" it away).
        """
        if self.ascii_image_active:
            return
        
        # Only initiate drops in currently lit non-drop cells
        image_top_cells_active = [cell for cell in self.image_top_cells if cell.is_lit and not cell.drop]
        if not image_top_cells_active:
            return

        # Increase drop probablity as less cells remain in the image top boundary (for better visual)
        drop_probability_start: float = Matrix.DROP_PROBABLITY / 30
        drop_probability_end: float = 0.05

        n_top_boundary_cells_initial = len(self.image_top_cells)
        n_top_boundary_cells_remaining = len(image_top_cells_active)

        # Increase wash drop probability in cubic progression for slow degradation in the beginning and fast in the end
        gradual_change = GradualChange(
            start_probability=drop_probability_start,
            end_probability=drop_probability_end,
            n_steps=n_top_boundary_cells_initial)
        drop_probablity = gradual_change.get_accelerating_probability(n_top_boundary_cells_initial - n_top_boundary_cells_remaining)

        for cell in image_top_cells_active:
            if random.random() > drop_probablity:
                continue
            drop_length = random.randint(self.MIN_DROP_LENGTH, self.MAX_DROP_LENGTH)
            cell.set_drop_head(drop_length)

    def change_rain_decelerating(self, target_drop_probability: float, change_time_elapsed_seconds: float, change_duration_seconds: float) -> None:
        """
        Change active drop probability based on time elapsed since the transition start.
        """
        gradual_change = GradualChange(
            start_probability=self.DROP_PROBABLITY,
            end_probability=target_drop_probability,
            n_steps=change_duration_seconds,
            exponent=2)
        
        self.active_drop_probability = gradual_change.get_decelerating_probability(change_time_elapsed_seconds)

    def spawn_glitches(self) -> None:
        """
        Spawn new glitches in random cells.
        """
        cells_to_glitch = []
        # Choose random cells to glitch.
        # One cell could be added several times.
        for row in self.rows:
            for cell in row:
                if random.random() < self.GLITCH_PROBABILITY:
                    cells_to_glitch += [cell]

        self.glitches += [Glitch(cell) for cell in cells_to_glitch]
    
    def apply_glitches(self) -> None:
        """
        Cycle through active glitches and apply their actions to cells.
        """
        # Remove glitches that have ran out of actions
        self.glitches = [glitch for glitch in self.glitches if glitch.action_queue]
        # When same cell has been randomly added to the glitch list several times, the later glitches action overrides the earlier one in every frame.
        for glitch in self.glitches:
            glitch.do_action()

    def set_message_texts(self, message_texts: list[str]) -> None:
        """
        Sets the text of the messages that can be revealed later.
        """
        self.message_texts: list = message_texts

    def spawn_message(self) -> None:
        """
        Selects a random message from available message texts, obfuscates it and places it in the matrix.
        At most one message per frame is spawned.
        """
        # A new message can only spawned if the current number of messages is smaller then the set number.
        if len(self.messages) >= self.N_CONCURRENT_MESSAGES:
            return
        message_text = random.choice(self.message_texts)

        # Obfuscate and pad message with spaces
        message_text_formatted = f"  {CharacterManipulation.get_obfuscated_text(message_text, self.MESSAGE_OBFUSCATION_PROBABILITY)}  "

        # Disregard the message if it can't be displayed completely
        if len(message_text_formatted) >= self.n_rows:
            return
        
        # Select a column that does not already have a message
        used_columns = [i_column for _, i_column in self.messages]
        available_columns = [i for i in range(self.n_columns) if i not in used_columns]
        i_column = random.choice(available_columns)

        # Select a starting row such that the message would fit the matrix
        i_start_row = random.choice(range(self.n_rows - len(message_text_formatted)))
        message_cells = [self.rows[i_row][i_column] for i_row in range(i_start_row, i_start_row + len(message_text_formatted))]

        message = Message([(cell, character) for cell, character in zip(message_cells, message_text_formatted)])
        self.messages += [(message, i_column)]

    def apply_messages(self) -> None:
        """
        Apply an action step on each message to reveal / hide the messages.
        Set messages for deletion with random probability.
        Disregard expired messages.
        """
        # Remove deleted messages only if they have completed all actions (i.e. are fully hidden)
        self.messages = [(message, i_column) for message, i_column in self.messages if not message.deleted or message.action]
        for message, _ in self.messages:
            message.do_action()
        # Delete, starting from the oldest message
        if self.messages and (random.random() < self.MESSAGE_REPLACE_PROBABLITY):
            self.messages[0][0].delete()


#####################
# Animation classes #
#####################

class TimingPlan:
    """
    Class for managing timed events during animation.
    Keeps track of time elapsed and indicates which events are currently due.
    """
    def __init__(self, **kwargs) -> None:
        self.events = {}
        for key, value in kwargs.items():
            self.events[key] = value
        self.timestamps = {}
    
    def set_timestamp_start(self, t: float) -> None:
        self.start_timestamp = t

    def set_timestamp_current(self, t: float) -> None:
        self.current_timestamp: float = t
        self.time_elapsed = self.current_timestamp - self.start_timestamp

    def set_timestamp_immutable(self, timestamp_name: str, t: float) -> None:
        """
        Function to set timestamps with chosen name. Timestamp for some name is only set once and is not updated.
        """
        if not self.timestamps.get(timestamp_name):
            self.timestamps[timestamp_name] = t

    def get_time_elapsed(self, timestamp_name: str = None) -> float:
        """
        Get time elapsed from a timestamp. If no timestamp name is provided, returns time elapsed from start timestamp.
        """
        reference_timestamp = self.timestamps.get(timestamp_name) or self.start_timestamp
        return self.current_timestamp - reference_timestamp

    def is_event_due(self, event_name: str) -> bool:
        """
        Returns whether time elapsed is past the start time of the input event.
        """
        return (event_start_time := self.events.get(event_name)) and event_start_time < self.time_elapsed


class Animation:
    """
    Class for orchestrating the matrix animation in terminal.
    """
    FRAME_SLEEP_PERIOD_SECONDS: float = 0.06            # Sets the speed of falling drops

    def __init__(self, matrix: Matrix) -> None:
        self.matrix = matrix
        self.is_running = False

    def print_frame(self) -> None:
        """
        Display frame in terminal.
        """
        # Print function "end" parameter is used to avoid adding newline to the end of the printed strings
        print(CharacterManipulation.return_to_top(), end="")    # Return to top
        print(self.matrix, end="", flush=True)                  # Flush screen and print the matrix
    
    def update_frame(self) -> None:
        """
        Blanket function, aggregating all necessary updates in an animation step (frame).
        """
        self.matrix.move_drops()
        self.matrix.spawn_drops()
        self.matrix.spawn_ascii_image_washing_drops()

        self.matrix.apply_glitches()
        self.matrix.spawn_glitches()

        self.matrix.apply_messages()
        self.matrix.spawn_message()

    def set_timing_plan(self, timing_plan: TimingPlan) -> None:
        self.timing_plan = timing_plan

    def apply_timing_plan(self) -> None:
        """
        Function that orchestrates timed changes in animation.
        """
        self.timing_plan.set_timestamp_current(time.time())

        # Display ascii image
        if self.timing_plan.is_event_due("start_ascii_image"):
            self.matrix.ascii_image_active = True
        
        # Stop rain
        if self.timing_plan.is_event_due("stop_rain"):
            # Idempotent actions that only have effect when first called
            self.matrix.rain_active = False
            self.timing_plan.set_timestamp_immutable("stop_rain", time.time())
            # Reduce drop probability to 0 over 30 seconds
            self.matrix.change_rain_decelerating(
                target_drop_probability=0,
                change_time_elapsed_seconds=self.timing_plan.get_time_elapsed("stop_rain"),
                change_duration_seconds=30)

        # Wash ascii image
        if self.timing_plan.is_event_due("wash_ascii_image"):
            self.matrix.ascii_image_active = False

        # Stop animation
        if self.timing_plan.is_event_due("total_run_time"):
            self.is_running = False

    def run(self) -> None:
        """
        Executes the animation.
        """
        self.is_running = True
        self.timing_plan.set_timestamp_start(time.time())
        while self.is_running:
            self.print_frame()
            self.update_frame()
            time.sleep(self.FRAME_SLEEP_PERIOD_SECONDS)
            if self.timing_plan:
                self.apply_timing_plan()


#######
# Run #
#######

if __name__ == "__main__":
    os.system("clear")
    time.sleep(5)

    # Event start times in seconds
    timing = dict(
        start_ascii_image = 200,
        stop_rain = 200 + 10,
        wash_ascii_image = 260,
        total_run_time = 260 + 40
        )
    
    while True:
        n_columns, n_rows = os.get_terminal_size()
        matrix = Matrix(n_rows, n_columns)

        with open("ascii_image.txt") as ascii_image_file:
            ascii_text = ascii_image_file.read()
        ascii_image = AsciiImage(ascii_text)
        matrix.set_ascii_image(ascii_image)

        with open("subliminal_messages.txt") as messages_file:
            message_texts = [message.strip() for message in messages_file.readlines()]
        matrix.set_message_texts(message_texts)

        animation = Animation(matrix)

        timing_plan = TimingPlan(**timing)
        animation.set_timing_plan(timing_plan)
        animation.run()
