from hanzi_font_deconstructor.common.TransformedStroke import TransformedStroke
from hanzi_font_deconstructor.common.generate_svg import generate_svg, get_stroke_attrs

def test_generate_svg(snapshot):
    strokes = [
        TransformedStroke(
            translate=(12,134),
            rotate=-3,
            skewX=3,
            skewY=-1,
            scale=(0.3, 1.7),
            path="M703 378l-64 21c-8 -47 -32 -118 -53 -173l61 -18c23 54 47 124 56 170z",
        ),
        TransformedStroke(
            translate=(-2,232),
            rotate=-3,
            skewX=-12,
            skewY=0,
            scale=(1, 0.9),
            path="M849 191l65 19c-32 75 -76 163 -113 220c-13 -9 -43 -23 -60 -30c40 -56 81 -137 108 -209z",
        )
    ]
    strokes_attrs = [get_stroke_attrs(stroke) for stroke in strokes]
    assert generate_svg(strokes_attrs, (-10, 0, 1010, 1000)) == snapshot