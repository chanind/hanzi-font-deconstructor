from dataclasses import asdict
from hanzi_font_deconstructor.common.generate_training_data import (
    STROKE_VIEW_BOX,
    get_training_input_svg_and_masks,
)
from hanzi_font_deconstructor.common.generate_svg import generate_svg, get_stroke_attrs
from os import path, makedirs
from pathlib import Path
import shutil
import argparse
import json

PROJECT_ROOT = Path(__file__).parents[2]
DEST_FOLDER = PROJECT_ROOT / "data"


parser = argparse.ArgumentParser(
    description="Generate training data for a model to deconstruct hanzi into strokes"
)
parser.add_argument("--max-strokes-per-img", default=5, type=int)
parser.add_argument("--total-images", default=50, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    # create and empty the dest folder
    if path.exists(DEST_FOLDER):
        shutil.rmtree(DEST_FOLDER)
    makedirs(DEST_FOLDER)
    makedirs(DEST_FOLDER / "sample_svgs")

    # create the data
    data = {
        "viewbox": STROKE_VIEW_BOX,
        "imgs": [],
    }
    for i in range(args.total_images):
        (img_svg, stroke_masks) = get_training_input_svg_and_masks(256)
        label = f"{i}-{len(stroke_masks)}"
        with open(DEST_FOLDER / "sample_svgs" / f"{label}.svg", "w") as img_file:
            img_file.write(img_svg)
        print(".")
    print("Done!")
