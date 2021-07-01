from dataclasses import dataclass
from .TransformedStroke import TransformedStroke
from typing import Sequence, Tuple


@dataclass
class StrokeAttrs:
    d: str
    transform: str


def generate_svg(
    strokes_attrs: Sequence[StrokeAttrs], viewbox=Tuple[int, int, int, int]
) -> str:
    svg_paths = []
    for attrs in strokes_attrs:
        svg_paths.append(f'<path d="{attrs.d}" transform="{attrs.transform}" />')
    viewboxStr = f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}"
    svgPathsStr = "\n".join(svg_paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" viewBox="{viewboxStr}">\n'
        f"{svgPathsStr}\n"
        "</svg>"
    )


def get_stroke_attrs(stroke: TransformedStroke) -> StrokeAttrs:
    return StrokeAttrs(
        d=stroke.path,
        transform=(
            " ".join(
                [
                    f"translate({stroke.translate[0]}, {stroke.translate[1]})",
                    f"rotate({stroke.rotate})",
                    f"skewX({stroke.skewX})",
                    f"skewY({stroke.skewY})",
                    f"scale({stroke.scale[0]}, {stroke.scale[1]})",
                ]
            )
        ),
    )
