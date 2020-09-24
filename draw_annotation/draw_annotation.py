from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple

def draw_annotation(image: Union[np.ndarray, str, Image.Image], annotation: Union[list, np.ndarray],
                    label: Union[bool, str] = None, color: Tuple[int, int, int, int] = None,
                    probs: list = None) -> np.ndarray:
    """
    Draw annotation (bbox or point(circle)) on image.
    Args:
        image:
        annotation: one or multiple bounding boxes/points.
        label: label to put on annotation if True and annotation shape == 2 (multiple) will enumerate instances.
        color: set color for instance

    Returns: image in numpy array format with drawn annotations

    """
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    fnt_height = fnt.getmetrics()  # ascent, descent
    point_size = 30
    rectangle_boarder_size = 10
    if color is None:
        color = 'red'
    point_draw_margin = np.array(((-point_size/2, -point_size/2), (point_size/2, point_size/2)))
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        if image.max() <= 1:
            image = (image*255).astype(np.uint8)
        if image.shape[0] == 3:
            image = image.transpose(1,2,0)
        image = Image.fromarray(image)
    # image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, 'RGBA')
    annotation = np.array(annotation)
    annotation = annotation.squeeze()
    if len(annotation.shape) == 1:  # single annotation sample
        if len(annotation) == 0:
            return np.array(image)
        if len(annotation) == 2:
            annotation = tuple((annotation + point_draw_margin).flatten())
            draw.ellipse(annotation, fill=color, outline=color)
        elif len(annotation) == 3:
            annotation = tuple((annotation[:-1] + point_draw_margin).flatten())
            draw.ellipse(annotation, fill=color, outline=color)
        elif len(annotation) == 4:
            draw.rectangle(tuple(annotation), width=rectangle_boarder_size, outline=color)
        elif len(annotation) == 5:
            draw.rectangle(tuple(annotation[:-1]), width=rectangle_boarder_size, outline=color)
        if label is not None:
            if isinstance(label, str):
                label_ = str(label)
                draw.text(annotation, label_, font=fnt, fill=color)
        if probs is not None:
            draw.text(annotation[::3] - np.array([0, sum(fnt_height)]), f'{probs:.2f}', font=fnt, fill=color)
    elif len(annotation.shape) == 2: # batch of annotation samples
        if probs is not None:
            assert len(probs) == len(annotation), 'Probabilities len not equal to annotation len'
        for num, annotation_ in enumerate(annotation):
            if len(annotation_) == 0:
                continue
            if len(annotation_) == 2:
                annotation_ = tuple((annotation_ + point_draw_margin).flatten())
                draw.ellipse(annotation_, fill=color, outline=color)
            elif len(annotation_) == 3:
                annotation_ = tuple((annotation_[:-1] + point_draw_margin).flatten())
                draw.ellipse(annotation_, fill=color, outline=color)
            elif len(annotation_) == 4:
                draw.rectangle(tuple(annotation_), width=rectangle_boarder_size, outline=color)
            elif len(annotation_) == 5:
                draw.rectangle(tuple(annotation_[:-1]), width=rectangle_boarder_size, outline=color)
            if label is not None:
                label_ = str(num)
                if isinstance(label, str):
                    label_ = str(label) + '_' + label_
                draw.text(annotation_[:2], label_, font=fnt, fill=color)
            if probs is not None:
                prob = probs[num]
                draw.text(annotation_[::3] - np.array([0, sum(fnt_height)]), f'{prob:.2f}', font=fnt, fill=color)

    return np.array(image)
