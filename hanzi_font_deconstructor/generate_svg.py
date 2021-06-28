from .TransformedStroke import TransformedStroke
from typing import Sequence, Tuple


def generate_svg(stroke_paths: Sequence[TransformedStroke], viewbox=Tuple[int, int, int, int]) -> str:
    svg_paths=[]
    for stroke_path in stroke_paths:
        svg_paths.append((
            f'<path fill="currentColor" d="{stroke_path.path}"'
            'transform="'
            f'  rotate({stroke_path.rotate})'
            f'  translate({stroke_path.translate[0]}, {stroke_path.translate[0]})'
            f'  skewX({stroke_path.skewX})'
            f'  skewY({stroke_path.skewY})'
            '" />'
        ))
    viewboxStr = f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}"
    svgPathsStr = "\n".join(svg_paths)
    return (
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" viewBox="{viewboxStr}">\n'
        f'{svgPathsStr}\n'
        '</svg>'
    )