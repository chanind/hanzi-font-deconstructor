from svgpathtools import parse_path
from typing import Tuple
from .TransformedStroke import TransformedStroke
from random import randint, uniform, gauss

def transform_stroke(stroke_pathstr: str, viewbox: Tuple[int, int, int, int]) -> TransformedStroke:
    path = parse_path(stroke_pathstr)
    strokeMinX, strokeMaxX, strokeMinY, strokeMaxY = path.bbox()
    vbMinX, vbMinY, vbWidth, vbHeight = viewbox
    vbMaxX = vbMinX + vbWidth
    vbMaxY = vbMinY + vbHeight

    translateX = randint(int(vbMinX - strokeMinX), int(vbMaxX - strokeMaxX))
    translateY = randint(int(vbMinY - strokeMinY), int(vbMaxY - strokeMaxY))
    rotate = uniform(-10, 10)
    skewX = uniform(-10, 10)
    skewY = uniform(-10, 10)
    scaleX = max(0.1, gauss(0.3, 0.2))
    scaleY = scaleX * uniform(0.9, 1.1)

    return TransformedStroke(
        translate=(translateX * min(scaleX, 1) * 0.9, translateY * min(scaleY, 1) * 0.9),
        rotate=rotate,
        scale=(scaleX, scaleY),
        skewX=skewX,
        skewY=skewY,
        path=stroke_pathstr,
    )
