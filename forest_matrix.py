import os
import random
import time
from typing import Callable


#########################
# Functions and Classes #
#########################

class CharacterManipulation:
    ESCAPE: str = "\x1b"
    SET_COLOUR_COMMAND: str = "38;5;"
    SETTING_START: str = "["
    SETTING_END: str = "m"
    HOME: str = "H"
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
        return f"{CharacterManipulation.ESCAPE}{CharacterManipulation.SETTING_START}{CharacterManipulation.SET_COLOUR_COMMAND}{colour256}{CharacterManipulation.SETTING_END}{character}"
    
    @staticmethod
    def return_to_top() -> str:
        return f"{CharacterManipulation.ESCAPE}{CharacterManipulation.SETTING_START}{CharacterManipulation.HOME}"
    
    @staticmethod
    def get_obfuscated_text(text: str, obfuscation_probability: float = 0.3) -> str:
        obfuscated_letters = []
        for letter in text:
            if (letter not in CharacterManipulation.OBFUSCATION_REGISTER.keys()) or (random.random() > obfuscation_probability):
                obfuscated_letters += [letter]
                continue
            obfuscated_letters += [random.choice(CharacterManipulation.OBFUSCATION_REGISTER[letter])]
        return "".join(obfuscated_letters)



class AsciiImage:
    BLANK_CHARACTER: str = " "
    NEWLINE_CHARACTER: str = "\n"

    def __init__(self, ascii_text: str) -> None:
        self.text: str = ascii_text

    def get_binary(self) -> list[list[bool]]:
        """
        Return a matrix (list or rows) where character locations are indicated by True values
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


class Drop:
    def __init__(self, length: int) -> None:
        self.length: int = length
        self.step: int = 1           # Parameter for possible extension: could make drops that move two cells in a cycle
    
    def get_colour(self, position_in_drop: int, bright_colours: int, lit_colours: list[int], fading_colours: list[int]) -> int:
        if position_in_drop == 0:
            return random.choice(bright_colours)
        
        # sequence of all colors
        colour_sequence = lit_colours + fading_colours
        # If the brightness bias is close to 1, the drop colour is biased towards brighter colours. I.e. the colours towards the beginning og the sequence
        brightness_bias: float = 0.7    # Values from 0 to 1
        i_colour = int(((len(colour_sequence) - 1) * position_in_drop / self.length - brightness_bias) // 1 + 1)
        return colour_sequence[i_colour]
    
    def get_next_position(self, position_in_drop: int) -> int:
        next_position = position_in_drop + self.step
        if next_position < self.length:
            return next_position
        return None


class Cell:
    # Colour codes in 256 colour system
    BRIGHT_COLOURS: list[int] = [231]           # whites
    LIT_COLOURS: list[int] = [48, 41, 35, 29]   # greens
    DIM_COLOURS: list[int] = [29, 22]           # dark greens
    FADING_COLOURS: list[int] = [238]           # grays
    INIVISIBLE_COLOUR: int = -1                 # black (color code 0 doesn't look good on screen, so we return a blank character instead)

    def __init__(self, character: str) -> None:
        self.character: str = character
        self.override_character: str = None

        self.is_lit: bool = False
        self.default_colour: int = random.choice(self.LIT_COLOURS)
        self.override_colour: int = None

        self.position_in_drop: int = 0          # Position starting from drop head. 0-based indexing.
        self.drop: Drop = None

        self.is_ascii_image: bool = False       # Cell is part of a 2d ascii "image"
        self.is_message: bool = False           # Cell is part of a vertical text "message"

    def __str__(self) -> str:
        if self.is_lit:
            active_character = self.get_active_character()
            active_colour = self.get_active_colour()
            return CharacterManipulation.get_coloured_character(active_character, active_colour)
        return CharacterManipulation.BLANK_CHARACTER

    def get_active_colour(self):
        if self.override_colour:
            return self.override_colour
        if self.drop:
            drop_colour = self.drop.get_colour(self.position_in_drop, self.BRIGHT_COLOURS, self.LIT_COLOURS, self.FADING_COLOURS)
            return drop_colour
        return self.default_colour

    def get_active_character(self):
        # black doesn't look good on screen, so we return a blank character instead
        if self.get_active_colour() == self.INIVISIBLE_COLOUR:
            return CharacterManipulation.BLANK_CHARACTER
        return self.override_character or self.character

    def set_drop_head(self, drop_length: int) -> None:
        self.position_in_drop = 0
        self.drop = Drop(drop_length)
        self.is_lit = True

    def move_drop(self, image_active: bool) -> None:
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
    def __init__(self, cell: Cell) -> None:
        self.cell: Cell = cell
        self.action_queue: list[Callable] = []

        # Don't glitch messages
        if cell.is_message:
            return

        self.action_queue += random.choice([self.burnout(), self.flicker_colour(), self.flicker_character()])
        self.action_queue += [self.clear]
        # Reverse, so it can be applied by pop()
        self.action_queue = list(reversed(self.action_queue))
    
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

    def flicker_colour(self) -> list:
        # Cycling through dim colours
        return random.randint(5, 20) * ([self.dim] + random.randint(5, 10) * [self.sleep])
    
    def flicker_character(self) -> list:
        # Change characters sequentially
        return random.randint(5, 10) * ([self.change_character] + 10 * [self.sleep])
    
    def burnout(self) -> list:
        # Go bright and then dark for a period, before reappearing again
        return [self.flash] + random.randint(1, 3) * [self.sleep] + random.randint(5, 20) * [self.invisible] + [self.change_character]
    
    def do_action(self) -> None:
        # Performs tne next action step and removes it from the queue
        action = self.action_queue.pop()
        action()


class Message:
    def __init__(self, cells: list[tuple[Cell, str]]) -> None:
        self.cells: list[tuple[Cell, str]] = cells
        # Set random order for applying actions
        self.action_queue: list[tuple[Cell, str]] = random.sample(self.cells, k=len(self.cells))
        # Variables for timing message action steps
        self.sleep: int = 0
        # Ongoing action
        self.action: tuple[Cell, str] = None
        # Deletion indicator
        self.deleted: bool = False
    
    @staticmethod
    def set_override_character(cell: Cell, character: str) -> None:
        # If character is None (i.e. deletion sequence), remove message indicator from cell
        cell.is_message = character is not None
        cell.override_character = character
    
    def do_action(self) -> None:
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
            self.sleep = random.randint(0, 2)
        # If the ongoing action has completed the sleep phase, set the message character
        self.set_override_character(*self.action)
        # Clear ongoing action
        self.action = None

    def delete(self) -> None:
        # Avoid restarting ongoing delete
        if not self.deleted:
            # Set random deletion order for cells
            self.action_queue = random.choices([(cell, None) for cell, _ in self.cells], k=len(self.cells))
            self.deleted = True


class Matrix:
    MIN_DROP_LENGTH: int = 4
    MAX_DROP_LENGTH: int = 25
    DROP_PROBABLITY: float = 0.01               # Drop probablity per column per step
    GLITCH_PROBABILITY: float = 0.0002          # Glitch probability per cell per step
    N_CONCURRENT_MESSAGES: int = 50             # Number of messages active at any time
    MESSAGE_REPLACE_PROBABLITY: float = 0.001   # Probablity that an existing message is deleted and another one spawned per cycle

    # Forestry related symbols: ϙ Ѧ ⍋ ⍦ ☙ ⚐ ⚘ ⚲ ⚶ ✿ ❀ ❦ ❧ ⲯ ⸙ 🙖 🜎
    CHARACTER_CODE_POINTS: list[int] = [985, 1126, 9035, 9062, 9753, 9872, 9880, 9906, 9910, 10047, 10048, 10086, 10087, 11439, 11801, 128598, 128782]
    AVAILABLE_CHARACTERS: list[int] = [chr(x) for x in CHARACTER_CODE_POINTS]

    FRAME_SLEEP_PERIOD_SECONDS: float = 0.07    # Sets the speed of falling drops

    def __init__(self, n_rows: int, n_columns: int) -> None:
        self.n_rows = n_rows
        self.n_columns = n_columns
        # List of rows consisting of cells
        self.rows: list[list[Cell]] = []
        # Populate the matrix
        for _ in range(self.n_rows):
            row = [Cell(character) for character in random.choices(self.AVAILABLE_CHARACTERS, k=self.n_columns)]
            self.rows.append(row)
        
        # Duplicated drop probability variable is set, because it changes when "stopping" the rain
        self.active_drop_probability: float = self.DROP_PROBABLITY
        self.rain_active: bool = True
        self.glitches: list[Glitch] = []
        # Tuples of (message, message column index)
        # Allows to control that every column only has one message
        self.messages: list[tuple[Message, int]] = []

    def __str__(self) -> str:
        # Get the string that is printed on screen
        return "".join("".join(str(cell) for cell in row) for row in self.rows)
    
    def set_ascii_image(self, ascii_image: AsciiImage) -> None:
        ascii_image_matrix = ascii_image.get_scaled_matrix(self.n_rows, self.n_columns)
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

    def set_message_texts(self, message_texts: list[str]) -> None:
        self.message_texts: list = message_texts

    def move_drops(self) -> None:
        # Iterate through rows starting from the bottom
        for i_row_above, row in reversed(list(enumerate(self.rows[1:]))):
            # Iterate through cells in the row
            for i_column, current_cell in enumerate(row):
                # Advance frame of each cell
                current_cell.move_drop(image_active = self.ascii_image_active)
                # If cell one row above is drop head, set cell as drop head
                cell_above = self.rows[i_row_above][i_column]
                if cell_above.drop and cell_above.position_in_drop == 0:
                    current_cell.set_drop_head(drop_length=cell_above.drop.length)
        
        # Advance frame for cells in first row
        for first_row_cell in self.rows[0]:
            first_row_cell.move_drop(image_active = self.ascii_image_active)

    def spawn_drops(self) -> None:
        for cell in self.rows[0]:
            if random.random() > self.active_drop_probability:
                continue

            drop_length = random.randint(self.MIN_DROP_LENGTH, self.MAX_DROP_LENGTH)
            cell.set_drop_head(drop_length)

    def spawn_ascii_image_washing_drops(self) -> None:
        if self.ascii_image_active:
            return
        
        # Only initiate drops in currently lit non-drop cells
        image_top_cells_active = [cell for cell in self.image_top_cells if cell.is_lit and not cell.drop]
        if not image_top_cells_active:
            return

        # Increase drop probablity as less cells remain in the image (for better visual)
        drop_probability_start: float = Matrix.DROP_PROBABLITY / 30
        drop_probability_end: float = 0.05

        n_message_columns_start = len(self.image_top_cells)
        n_message_columns_remaining = len(image_top_cells_active)

        # Increase wash drop probability in cubic progression as less columns remain for slow degradation in the beginning and fast end
        stop_function = lambda x: drop_probability_start + (drop_probability_end - drop_probability_start) * (1 - x / n_message_columns_start)**3
        drop_probablity = stop_function(n_message_columns_remaining)

        for cell in image_top_cells_active:
            if random.random() > drop_probablity:
                continue
            drop_length = random.randint(self.MIN_DROP_LENGTH, self.MAX_DROP_LENGTH)
            cell.set_drop_head(drop_length)

    def spawn_glitches(self) -> None:
        cells_to_glitch = []
        # Choose random cells to glitch
        for row in self.rows:
            for cell in row:
                if random.random() < self.GLITCH_PROBABILITY:
                    cells_to_glitch += [cell]

        self.glitches += [Glitch(cell) for cell in cells_to_glitch]
    
    def apply_glitches(self) -> None:
        # Remove glitches that have ran out of actions
        self.glitches = [glitch for glitch in self.glitches if glitch.action_queue]
        for glitch in self.glitches:
            glitch.do_action()

    def spawn_message(self) -> None:
        if len(self.messages) >= self.N_CONCURRENT_MESSAGES:
            return
        message_text = random.choice(self.message_texts)
        # Obfuscate and pad with spaces
        message_text_formatted = f" {CharacterManipulation.get_obfuscated_text(message_text)} "
        if len(message_text_formatted) >= self.n_rows:
            return
        # Select a column that does not already have a message
        used_columns = [i_column for _, i_column in self.messages]
        available_columns = [i for i in range(self.n_columns) if i not in used_columns]
        i_column = random.choice(available_columns)
        # Select a row such that the message would fit the column
        i_start_row = random.choice(range(self.n_rows - len(message_text_formatted)))
        message_cells = [self.rows[i_row][i_column] for i_row in range(i_start_row, i_start_row + len(message_text_formatted))]
        message = Message([(cell, character) for cell, character in zip(message_cells, message_text_formatted)])
        self.messages += [(message, i_column)]

    def apply_messages(self) -> None:
        # Remove deleted messages that have completed all actions
        self.messages = [(message, i_column) for message, i_column in self.messages if not message.deleted or message.action]
        for message, _ in self.messages:
            message.do_action()
        # Delete, starting from the oldest message
        if self.messages and (random.random() < self.MESSAGE_REPLACE_PROBABLITY):
            self.messages[0][0].delete()

    def print_frame(self) -> None:
        # use print end parameter to avoid adding newline to the end
        # Return to top and flush the screen on every frame
        print(CharacterManipulation.return_to_top(), end="")
        print(self, end="", flush=True)
        time.sleep(self.FRAME_SLEEP_PERIOD_SECONDS)

    def run(self) -> None:
        # Execution timing
        start_timestamp: float = time.time()
        start_ascii_image_seconds: int = 20
        stop_rain_seconds: int = 28
        wash_ascii_image_seconds: int = 80
        cycle_end_seconds: int = 120

        while (time.time() - start_timestamp) < cycle_end_seconds:
            # Print current state and then advance the frame
            self.print_frame()

            self.move_drops()
            self.spawn_drops()

            self.apply_glitches()
            self.spawn_glitches()

            self.apply_messages()
            self.spawn_message()

            # Switches for different stages of the animation
            if (time.time() - start_timestamp) > start_ascii_image_seconds:
                self.ascii_image_active = True

            if (time.time() - start_timestamp) > stop_rain_seconds:
                self.rain_active = False

            if (time.time() - start_timestamp) > wash_ascii_image_seconds:
                self.ascii_image_active = False
                self.spawn_ascii_image_washing_drops()
            
            if not self.rain_active:
                stop_transition_duration_seconds = 30
                seconds_past_stop_command = time.time() - start_timestamp - stop_rain_seconds
                # Reduce drop probability in cubic progression to stop rain gradually
                stop_function = lambda x: self.DROP_PROBABLITY * abs(min(0, (x / stop_transition_duration_seconds - 1)**3))
                self.active_drop_probability = stop_function(seconds_past_stop_command)


#######
# Run #
#######

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

    matrix.run()
