from .generate_svg import generate_svg, get_stroke_attrs
from .transform_stroke import transform_stroke
from .transform_stroke import transform_stroke
from .svg_to_pil import svg_to_pil
from os import path
from pathlib import Path
import random
import re
import torch
from torchvision import transforms


PROJECT_ROOT = Path(__file__).parents[2]
GLYPH_SVGS_DIR = PROJECT_ROOT / "noto_glyphs"

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

tensorify = transforms.ToTensor()


def img_to_greyscale_tensor(img):
    return tensorify(img)[3, :, :]


def get_mask_span(mask):
    "return a tuple of (horizontal span, vertical span)"
    horiz_max_vals = torch.max(mask, 0).values
    vert_max_vals = torch.max(mask, 1).values
    min_x = torch.argmax(horiz_max_vals).item()
    max_x = len(horiz_max_vals) - torch.argmax(torch.flip(horiz_max_vals, [0])).item()
    min_y = torch.argmax(vert_max_vals).item()
    max_y = len(vert_max_vals) - torch.argmax(torch.flip(vert_max_vals, [0])).item()
    return (max_x - min_x, max_y - min_y)


def is_stroke_good(mask, existing_masks) -> bool:
    # if this is the first stroke, then anything is fine
    if len(existing_masks) == 0:
        return True

    # TODO: this is probably really slow, might need to speed this up somehow
    mask_size = torch.sum(mask).item()
    # mask_span = get_mask_span(mask)
    for existing_mask in existing_masks:
        existing_mask_size = torch.sum(existing_mask).item()
        overlaps = torch.where(existing_mask + mask >= 2, 1, 0)
        overlaps_size = torch.sum(overlaps).item()
        if overlaps_size == 0:
            # if this is the second stroke, ensure there's an overlap
            # we should ensure there's at least 1 overlap per training sample
            return len(existing_masks) > 1
        # if the overlap is a large amount of either stroke, this is a bad stroke
        if overlaps_size / existing_mask_size > 0.25:
            return False
        if overlaps_size / mask_size > 0.25:
            return False
        # overlaps_span = get_mask_span(overlaps)
        # existing_mask_span = get_mask_span(existing_mask)
        # # if the overlap is a large amount of the span of either stroke, this is a bad stroke
        # if max(overlaps_span) / max(existing_mask_span) > 0.4:
        #     return False
        # if max(overlaps_span) / max(mask_span) > 0.4:
        #     return False
    return True


MASK_THRESHOLD = 0.3


def get_training_input_svg_and_masks(size_px):
    num_strokes = random.randint(3, 4)
    with torch.no_grad():
        strokes_attrs = []
        stroke_masks = []
        while len(strokes_attrs) < num_strokes:
            pathstr = random.choice(SINGLE_STROKE_PATHS)
            stroke = transform_stroke(pathstr, STROKE_VIEW_BOX)
            stroke_attrs = get_stroke_attrs(stroke)
            stroke_svg = generate_svg([stroke_attrs], STROKE_VIEW_BOX)
            stroke_img = svg_to_pil(stroke_svg, size_px, size_px)
            stroke_tensor = img_to_greyscale_tensor(stroke_img)
            stroke_mask = torch.where(stroke_tensor > MASK_THRESHOLD, 1, 0)

            if is_stroke_good(stroke_mask, stroke_masks):
                strokes_attrs.append(stroke_attrs)
                stroke_masks.append(stroke_mask)
        input_svg = generate_svg(strokes_attrs, STROKE_VIEW_BOX)
    return (input_svg, stroke_masks)


def get_training_input_and_mask_tensors(size_px=256):
    with torch.no_grad():
        input_svg, stroke_masks = get_training_input_svg_and_masks(size_px)

        input_img = svg_to_pil(input_svg, size_px, size_px)
        input_tensor = img_to_greyscale_tensor(input_img)
        mask_sums = torch.zeros(input_tensor.shape, dtype=torch.long)
        for stroke_mask in stroke_masks:
            mask_sums += stroke_mask

        # collapse all overlaps of more than 2 items into a single "overlap" class
        mask = torch.where(mask_sums > 2, 2, mask_sums)
        return (input_tensor.unsqueeze(0), mask)
