from hanzi_font_deconstructor.common.generate_training_data import (
    get_mask_bounds,
    get_training_input_svg_and_masks,
)
from os import path, makedirs
from pathlib import Path
from json import dump
import shutil
import argparse
import cv2
from tqdm import tqdm
import numpy as np

from hanzi_font_deconstructor.common.svg_to_pil import svg_to_pil

PROJECT_ROOT = Path(__file__).parents[2]
DEST_FOLDER = PROJECT_ROOT / "data" / "coco"


parser = argparse.ArgumentParser(
    description="Generate training data for a model to deconstruct hanzi into strokes"
)
parser.add_argument("--max-strokes-per-img", default=5, type=int)
parser.add_argument("--total-images", default=50, type=int)
parser.add_argument("--image-size", default=256, type=int)
parser.add_argument("--validation-portion", default=0.05, type=float)
args = parser.parse_args()

if __name__ == "__main__":
    # create and empty the dest folder
    if path.exists(DEST_FOLDER):
        shutil.rmtree(DEST_FOLDER)
    makedirs(DEST_FOLDER)
    makedirs(DEST_FOLDER / "images")

    # create the data

    annotation_counter = 0
    img_counter = 0
    for stage, total_images in [
        ("train", args.total_images),
        ("val", int(args.validation_portion * args.total_images)),
    ]:
        categories = [{"supercategory": "none", "name": "stroke", "id": 0}]
        coco_contents = {"categories": categories, "images": [], "annotations": []}
        print(f"generating {stage} dataset")
        for _ in tqdm(range(total_images)):
            (img_svg, stroke_masks) = get_training_input_svg_and_masks(args.image_size)
            img_filename = f"images/{img_counter}-{len(stroke_masks)}.png"
            img_elem = {
                "file_name": img_filename,
                "height": args.image_size,
                "width": args.image_size,
                "id": img_counter,
            }
            coco_contents["images"].append(img_elem)
            for stroke_mask in stroke_masks:
                min_x, max_x, min_y, max_y = get_mask_bounds(stroke_mask)
                width = max_x - min_x
                height = max_y - min_y
                area = stroke_mask.sum().item()
                # poly = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

                cv_stroke_mask = stroke_mask.unsqueeze(-1).numpy().astype(np.uint8)
                contours = cv2.findContours(
                    cv_stroke_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                poly = contours[0][0][:, 0, :].flatten().tolist()

                annot_elem = {
                    "id": annotation_counter,
                    "bbox": [float(min_x), float(min_y), float(width), float(height)],
                    "segmentation": [poly],
                    "image_id": img_counter,
                    "ignore": 0,
                    "category_id": 0,
                    "iscrowd": 0,
                    "area": float(area),
                }
                coco_contents["annotations"].append(annot_elem)
                annotation_counter += 1

            img = svg_to_pil(img_svg, args.image_size, args.image_size)
            img.save(DEST_FOLDER / img_filename, format="png")
            img_counter += 1
        with open(DEST_FOLDER / f"{stage}.json", "w") as coco_data_file:
            dump(coco_contents, coco_data_file, ensure_ascii=False)
    print("Done!")
