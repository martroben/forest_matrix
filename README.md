# Forest matrix
The script starts a matrix-like rain animation in terminal. It's a refactored version of an [original](https://github.com/principle105/matrix_rain/tree/main/matrix_rain) by Github user [principle105](original_creator_license.md).

**To run**:
```shell
git clone https://github.com/martroben/ad_hoc/

cd ad_hoc/forest_matrix
python3 forest_matrix.py
```

## Working principle
The script determines the current terminal size (in character rows x character columns) and starts overprinting strings that have the same size as the terminal. These alternating strings are the frames of the "movie".

Different cells in the matrix are assigned characters and colours with [ANSI escape codes](https://en.wikipedia.org/wiki/ANSI_escape_code) to create an animation.

Positions in the matrix are assigned characters during initialisation. Characters are chosen randomly for a pre-determined set. The drops just reveal and hide these characters. The timing and length of the drops is probabilistic.

## Features
Apart from the drops, the matrix animation also includes "glitches", "messages", static ascii image and changes in rainfall.

### Glitches
Random cells in matrix either light up and go dark, flicker or change characters repeatedly. The glitches are only visible if the glitching cell is currently lit up by a drop or a static ascii image.

### Messages
Vertical messages appear and disappear in the matrix and are visible when a falling drop lights them up. Messages are randomly selected from the lines in [subliminal_messages.txt](subliminal_messages.txt). Messages are obfuscated by randomly substituting visually similar characters to make them look more Matrix-y.

### Static ascii image
An ascii image is loaded from [ascii_image.txt](ascii_image.txt). The (non-whitespace) characters of the file are replaced by matrix characters. It is placed in the center of the matrix and gradually revealed by falling drops. Later the image is washed away by spawning drops from the top edge of the image.

### Changes in rainfall
The rain can be gradually stopped by reducing the active probablity of new drops to zero.

## Notes
- 1920 x 1080 (full HD) resolution corresponds to 56 rows x 209 columns.
- use `python3 -c "import os; print(os.get_terminal_size())"` in terminal to get the current terminal dimensions.
- Windows Terminal tips:
    - use Settings > Rendering > Use software rendering: "on" for smoother animation.
    - use Settings > Defaults > Cursor shape: "Vintage" and Cursor height: 1 to remove cursor flicker.
    - Ascii image [generation tool](https://seotoolbelt.co/tools/ascii-art-generator/#text-list-tab) was used to create the sample image.
- Obfuscation character replacements were (mostly) pulled from this [obfuscator tool](https://obfuscator.uo1.net/).
- Drop speed can be adjusted by `FRAME_SLEEP_PERIOD_SECONDS` in `Animation` class.
- There are a lot of other settings available via the `Matrix` and `Cell` class variables.
- I used Ubuntu screen recording in full-screen Gnome shell to capture a good quality video. Then [Handbrake](https://handbrake.fr/) video editor on Win11 to convert from webm to mp4.
