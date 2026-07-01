from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from configs.loader import CLASS_COLORS, CLASS_NAMES_RU


DEFAULT_COLOR = "#ff3333"


def get_font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def get_text_size(draw, text, font):
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def rgb_to_hex(color):
    if isinstance(color, str):
        return color

    if isinstance(color, list) or isinstance(color, tuple):
        return "#{:02x}{:02x}{:02x}".format(*color)

    return DEFAULT_COLOR


def draw_single_box(draw, bbox, label, color, line_width, font):
    xmin, ymin, xmax, ymax = bbox

    draw.rectangle(
        [(xmin, ymin), (xmax, ymax)],
        outline=color,
        width=line_width,
    )

    text_width, text_height = get_text_size(draw, label, font)

    image_width = draw.im.size[0]
    label_x = min(xmin, image_width - text_width - 8)
    label_x = max(0, label_x)

    label_y = max(0, ymin - text_height - 6)

    draw.rectangle(
        [
            (label_x, label_y),
            (label_x + text_width + 8, label_y + text_height + 6),
        ],
        fill=color,
    )

    draw.text(
        (label_x + 4, label_y + 3),
        label,
        fill="white",
        font=font,
    )


def save_manifest_annotation(
    image_path,
    objects,
    output_path,
    line_width=2,
    font_size=18,
    use_ru_labels=True,
):
    image_path = Path(image_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = get_font(font_size)

    for obj in objects:
        class_name = obj["class_name"]
        bbox_data = obj["bbox"]

        bbox = [
            bbox_data["xmin"],
            bbox_data["ymin"],
            bbox_data["xmax"],
            bbox_data["ymax"],
        ]

        label = CLASS_NAMES_RU.get(class_name, class_name) if use_ru_labels else class_name
        color = rgb_to_hex(CLASS_COLORS.get(class_name, DEFAULT_COLOR))

        draw_single_box(
            draw=draw,
            bbox=bbox,
            label=label,
            color=color,
            line_width=line_width,
            font=font,
        )

    image.save(output_path)


def save_prediction_annotation(
    image_path,
    predictions,
    output_path,
    line_width=2,
    font_size=18,
    use_ru_labels=True,
    show_confidence=True,
):
    image_path = Path(image_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = get_font(font_size)

    for prediction in predictions:
        class_name = prediction["class_name"]
        bbox = prediction["bbox"]

        label = CLASS_NAMES_RU.get(class_name, class_name) if use_ru_labels else class_name

        if show_confidence and "confidence" in prediction:
            label = f"{label} {prediction['confidence']:.2f}"

        color = rgb_to_hex(CLASS_COLORS.get(class_name, DEFAULT_COLOR))

        draw_single_box(
            draw=draw,
            bbox=bbox,
            label=label,
            color=color,
            line_width=line_width,
            font=font,
        )

    image.save(output_path)