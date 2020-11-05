from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
from .color import colors as default_colors


def draw_pose(image: Union[np.ndarray, str, Image.Image], annotation: Union[list, np.ndarray],
                    label: Union[bool, str] = None,
                    probs: list = None, point_size: int = 30, rectangle_boarder_size: int = 10,
                    font_size: int = 40, transparency: int = 200, skeleton_width: int = 10) -> np.ndarray:
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
        [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    fnt_height = fnt.getmetrics()  # ascent, descent
    color = [tuple(c + [transparency]) for c in default_colors]
    point_draw_margin = np.array(((-point_size/2, -point_size/2), (point_size/2, point_size/2)))
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        if image.max() <= 1:
            image = (image*255).astype(np.uint8)
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        image = Image.fromarray(image)
    # image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, 'RGBA')
    annotation = np.array(annotation)
    annotation = annotation.squeeze()
    ## draw joints
    for num, annotation_ in enumerate(annotation):
        if len(annotation_) == 0:
            continue
        if len(annotation_) == 2:
            annotation_ = tuple((annotation_ + point_draw_margin).flatten())
            draw.ellipse(annotation_, fill=color[num], outline=tuple(color[num]))
    ## draw skeleton
    for sk in skeleton:
        point1 = annotation[sk[0]]
        point2 = annotation[sk[1]]
        color = tuple(color[sk[0]] + [transparency])
        draw.line(point1 + point2, fill=color, width=skeleton_width)
    return np.array(image)
