from svgpathtools import parse_path
from typing import Tuple
from .TransformedStroke import TransformedStroke
from random import randint, uniform, gauss


def transform_stroke(
    stroke_pathstr: str, viewbox: Tuple[int, int, int, int]
) -> TransformedStroke:
    path = parse_path(stroke_pathstr)
    strokeMinX, strokeMaxX, strokeMinY, strokeMaxY = path.bbox()
    vbMinX, vbMinY, vbWidth, vbHeight = viewbox
    vbMaxX = vbMinX + vbWidth
    vbMaxY = vbMinY + vbHeight
    strokeMidX = (strokeMaxX + strokeMinX) / 2
    strokeMidY = (strokeMaxY + strokeMinY) / 2

    rotate = gauss(0, 3)
    skewX = gauss(0, 2)
    skewY = gauss(0, 2)
    scaleX = min(1.05, max(0.5, gauss(0.8, 0.2)))
    scaleY = scaleX * uniform(0.95, 1.05)

    # from https://stackoverflow.com/a/11671373
    baseTranslateX = (1 - scaleX) * strokeMidX
    baseTranslateY = (1 - scaleY) * strokeMidY
    translateX = baseTranslateX + randint(
        int(vbMinX - strokeMinX), int(vbMaxX - strokeMaxX)
    )
    translateY = baseTranslateY + randint(
        int(vbMinY - strokeMinY), int(vbMaxY - strokeMaxY)
    )

    return TransformedStroke(
        translate=(translateX, translateY),
        rotate=rotate,
        scale=(scaleX, scaleY),
        skewX=skewX,
        skewY=skewY,
        path=stroke_pathstr,
    )
