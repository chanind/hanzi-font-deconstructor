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


def get_training_img_strokes(max_strokes=5):
    num_strokes = random.randint(1, max_strokes)
    rand_stroke_pathstrings = [
        random.choice(SINGLE_STROKE_PATHS) for _ in range(num_strokes)
    ]
    transformed_strokes = [
        transform_stroke(pathstr, STROKE_VIEW_BOX)
        for pathstr in rand_stroke_pathstrings
    ]
    return transformed_strokes


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
    return tensorify(img)[3, :, :].unsqueeze(0)


def get_training_input_and_mask_tensors(max_strokes=5, size_px=512, mask_threshold=0.3):
    with torch.no_grad():
        strokes = get_training_img_strokes(max_strokes)
        strokes_attrs = [get_stroke_attrs(stroke) for stroke in strokes]
        input_svg = generate_svg(strokes_attrs, STROKE_VIEW_BOX)
        input_img = svg_to_pil(input_svg, size_px, size_px)
        input_tensor = img_to_greyscale_tensor(input_img)

        stroke_svgs = [
            generate_svg([stroke_attrs], STROKE_VIEW_BOX)
            for stroke_attrs in strokes_attrs
        ]
        stroke_imgs = [
            svg_to_pil(stroke_svg, size_px, size_px) for stroke_svg in stroke_svgs
        ]
        stroke_tensors = [
            img_to_greyscale_tensor(stroke_img) for stroke_img in stroke_imgs
        ]
        stroke_masks = [
            torch.where(stroke_tensor > mask_threshold, 1, 0)
            for stroke_tensor in stroke_tensors
        ]
        mask = torch.zeros(input_tensor.shape, dtype=torch.int)
        for stroke_mask in stroke_masks:
            mask += stroke_mask

        return (input_tensor, mask)


class RandomStrokesDataset(torch.utils.data.IterableDataset):
    def __init__(self, total_samples: int, max_strokes=5, size_px=512):
        super()
        self.total_samples = total_samples
        self.max_strokes = max_strokes
        self.size_px = size_px

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        total_per_worker = int(self.total_samples / num_workers)
        for _ in range(total_per_worker):
            yield get_training_input_and_mask_tensors(
                max_strokes=self.max_strokes, size_px=self.size_px
            )
