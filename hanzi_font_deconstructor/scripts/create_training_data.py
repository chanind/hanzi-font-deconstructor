from hanzi_font_deconstructor.common.generate_svg import generate_svg
from hanzi_font_deconstructor.common.transform_stroke import transform_stroke
from os import path, makedirs
from pathlib import Path
import shutil
import random
import re
import argparse

PROJECT_ROOT = Path(__file__).parents[2]

GLYPH_SVGS_DIR = PROJECT_ROOT / "noto_glyphs"
DEST_FOLDER = PROJECT_ROOT / "data"

# https://en.wikipedia.org/wiki/Stroke_(CJK_character)
SINGLE_STROKE_CHARS = [
    "一",
    "乙",
    "丨",
    "丶",
    "丿",
    "乀",
    "乁",
    "乚",
    "乛",
    "亅",
    "𠃊",
    "𠃋",
    "𠃌",
    "𠃍",
    "𠃑",
    "𠄌",
    "㇐",
    "㇀",
    "㇖",
    "㇇",
    "㇕",
    "㇆",
    "㇊",
    "㇅",
    "㇍",
    "㇈",
    "㇠",
    "㇎",
    "㇋",
    "㇌",
    "㇡",
    "㇑",
    "㇚",
    "㇙",
    "㇗",
    "㇄",
    "㇘",
    "㇟",
    "㇞",
    "㇉",
    "㇒",
    "㇓",
    "㇢",
    "㇜",
    "㇛",
    "㇔",
    "㇏",
    "㇝",
    "㇂",
    "㇃",
    "㇁",
]


STROKE_VIEW_BOX = (-10, 0, 1010, 1000)
MISC_SINGLE_STROKE_PATHS = [
    "M884 65l34 62c-131 40 -349 62 -523 72c-2 -18 -10 -44 -18 -61c173 -12 387 -36 507 -73z",
    "M542 409 l-60 26c-14 -47 -46 -122 -74 -178l57 -22c30 56 63 127 77 174z",
    "M703 378l-64 21c-8 -47 -32 -118 -53 -173l61 -18c23 54 47 124 56 170z",
    "M849 191l65 19c-32 75 -76 163 -113 220c-13 -9 -43 -23 -60 -30c40 -56 81 -137 108 -209z",
    "M253 417v359c21 9 43 27 78 46c63 34 150 40 258 40c113 0 269 -7 370 -19c-10 22 -22 60 -24 83 c-78 5 -248 10 -349 10c-116 0 -202 -10 -270 -46c-41 -22 -73 -50 -95 -50c-34 0 -82 52 -129 108l-53 -71c48 -46 99 -84 142 -100v-290h-128v-70h200z",
    "M267 239l-62 45c-27 -45 -88 -111 -142 -158l58 -39c53 43 117 108 146 152z",
    "M268 753l113 -80c6 22 18 51 26 66c-186 137 -214 160 -228 178c-8 -18 -28 -51 -43 -66c21 -14 58 -48 58 -92v-331h-145v-72h219v397z",
]


def get_file_for_char(char):
    code = hex(ord(char))[2:]
    return path.join(GLYPH_SVGS_DIR, f"{code}.svg")


SINGLE_STROKE_CHAR_PATHS = []
path_extractor = re.compile(r'\bd="([^"]+)"')
for char in SINGLE_STROKE_CHARS:
    char_file = get_file_for_char(char)
    with open(char_file, "r") as contents:
        char_svg = contents.read().replace("\n", "")
    path_match = path_extractor.search(char_svg)
    if not path_match:
        raise Exception(f"No SVG path found in char svg: {char}")
    SINGLE_STROKE_CHAR_PATHS.append(path_match[1])

SINGLE_STROKE_PATHS = MISC_SINGLE_STROKE_PATHS + SINGLE_STROKE_CHAR_PATHS

TARGET_IMG_SIZE_PX = 512


def get_training_img_strokes(max_strokes=5):
    num_strokes = random.randint(1, max_strokes)
    rand_stroke_pathstrings = [random.choice(SINGLE_STROKE_PATHS) for _ in range(num_strokes)]
    transformed_strokes = [transform_stroke(pathstr, STROKE_VIEW_BOX) for pathstr in rand_stroke_pathstrings]
    return transformed_strokes

parser = argparse.ArgumentParser(description='Generate training data for a model to deconstruct hanzi into strokes')
parser.add_argument('--max-strokes-per-img', default=5, type=int)
parser.add_argument('--total-images', default=10000, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    # create and empty the dest folder
    if path.exists(DEST_FOLDER):
        shutil.rmtree(DEST_FOLDER)
    makedirs(DEST_FOLDER)

    # create the data
    for i in range(args.total_images):
        training_strokes = get_training_img_strokes(args.max_strokes_per_img)
        label = f"{i}-{len(training_strokes)}"
        with open(DEST_FOLDER / f"{label}.svg", "w") as img_file:
            img_file.write(generate_svg(training_strokes, STROKE_VIEW_BOX))
        if i % 1000 == 0:
            print(f'written {i} / {args.total_images}')
    print('Done!')



