from io import BytesIO
from cairosvg import svg2png
from PIL import Image

def svg_to_pil(svg: str, width: int, height: int):
    out = BytesIO()
    svg2png(bytestring=svg.encode('utf-8'), parent_width=width, parent_height=height, write_to=out)
    return Image.open(out)