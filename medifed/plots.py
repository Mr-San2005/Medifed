"""Small PIL-based plotting helpers for summary charts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (245, 247, 248)
INK = (27, 42, 51)
MUTED = (91, 107, 117)
BORDER = (216, 224, 228)
PRIMARY = (15, 107, 110)
AMBER = (199, 125, 10)
HOSPITAL_COLORS = {
    1: (15, 107, 110),
    2: (63, 94, 154),
    3: (163, 83, 43),
}


def _make_font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _safe_error(value):
    return None if value is None else float(value)


def grouped_bar_chart(
    output_path: str | Path,
    labels: Iterable[str],
    series: list[dict],
    width: int = 1600,
    height: int = 900,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    """Create a grouped bar chart with optional error bars.

    series is a list of dictionaries, each containing `label`, `values`, and
    optional `errors` lists.
    """
    labels = list(labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(img)
    font = _make_font(12)

    left, top, right, bottom = 140, 100, width - 80, height - 90
    plot_w = right - left
    plot_h = bottom - top
    draw.rectangle([left, top, right, bottom], outline=BORDER, width=2)

    if title:
        draw.text((left, 25), title, fill=INK, font=font)
    if xlabel:
        draw.text((left + plot_w // 2 - 40, height - 35), xlabel, fill=INK, font=font)
    if ylabel:
        draw.text((20, top + plot_h // 2), ylabel, fill=INK, font=font)

    all_values = []
    for item in series:
        all_values.extend(item.get("values", []))
    max_value = max(all_values) if all_values else 1.0
    if max_value <= 0:
        max_value = 1.0
    y_tick = max_value / 5

    for tick in range(6):
        y = top + plot_h - tick * (plot_h / 5)
        value = tick * y_tick
        draw.text((left - 90, y - 7), f"{value:.2f}", fill=MUTED, font=font)
        draw.line((left, y, right, y), fill=BORDER, width=1)

    bar_count = len(labels)
    group_w = plot_w / bar_count
    group_gap = 0.18 * group_w
    bar_w = (group_w - group_gap * (len(series) + 1)) / max(len(series), 1)

    for idx, label in enumerate(labels):
        x_center = left + (idx + 0.5) * group_w
        x0 = x_center - (len(series) * bar_w + (len(series) - 1) * group_gap) / 2
        draw.text((x0, bottom + 12), str(label), fill=INK, font=font)

        for series_idx, item in enumerate(series):
            value = float(item["values"][idx])
            color = item.get("color", list(HOSPITAL_COLORS.values())[series_idx % len(HOSPITAL_COLORS)])
            x1 = x0 + series_idx * (bar_w + group_gap)
            y_top = top + plot_h - (value / max_value) * plot_h
            draw.rounded_rectangle(
                [x1, y_top, x1 + bar_w, bottom],
                radius=8,
                fill=color,
                outline=INK,
                width=1,
            )
            error = _safe_error(item.get("errors", [None] * len(labels))[idx])
            if error is not None:
                err_top = top + plot_h - ((value + error) / max_value) * plot_h
                err_bottom = top + plot_h - ((value - error) / max_value) * plot_h
                draw.line((x1 + bar_w / 2, err_top, x1 + bar_w / 2, err_bottom), fill=INK, width=2)
                draw.line((x1 + bar_w / 2 - 6, err_top, x1 + bar_w / 2 + 6, err_top), fill=INK, width=2)
                draw.line((x1 + bar_w / 2 - 6, err_bottom, x1 + bar_w / 2 + 6, err_bottom), fill=INK, width=2)

    legend_y = 40
    for item in series:
        draw.rectangle([right - 240, legend_y, right - 220, legend_y + 12], fill=item.get("color", PRIMARY))
        draw.text((right - 210, legend_y - 4), item["label"], fill=INK, font=font)
        legend_y += 25

    img.save(output_path, format="PNG")
    return output_path


def line_chart(
    output_path: str | Path,
    x_values: Iterable[int | float],
    series: list[dict],
    width: int = 1600,
    height: int = 900,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    """Create a line chart with optional error bars."""
    x_values = list(x_values)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(img)
    font = _make_font(12)

    left, top, right, bottom = 140, 100, width - 80, height - 90
    plot_w = right - left
    plot_h = bottom - top
    draw.rectangle([left, top, right, bottom], outline=BORDER, width=2)

    if title:
        draw.text((left, 25), title, fill=INK, font=font)
    if xlabel:
        draw.text((left + plot_w // 2 - 40, height - 35), xlabel, fill=INK, font=font)
    if ylabel:
        draw.text((20, top + plot_h // 2), ylabel, fill=INK, font=font)

    all_values = []
    for item in series:
        all_values.extend(item.get("values", []))
    if not all_values:
        img.save(output_path, format="PNG")
        return output_path

    max_value = max(all_values) if all_values else 1.0
    min_value = min(all_values) if all_values else 0.0
    if max_value == min_value:
        max_value += 0.1
        min_value -= 0.1
    y_top = max_value + 0.05 * (max_value - min_value)
    y_bottom = min_value - 0.05 * (max_value - min_value)

    for tick in range(5):
        y = top + plot_h - (tick / 4) * plot_h
        val = y_bottom + (y_top - y_bottom) * ((plot_h - (y - top)) / plot_h)
        draw.text((left - 80, y - 8), f"{val:.2f}", fill=MUTED, font=font)
        draw.line((left, y, right, y), fill=BORDER, width=1)

    x_count = max(len(x_values), 2)
    for x_index, x_value in enumerate(x_values):
        x = left + (x_index / (x_count - 1)) * plot_w
        draw.text((x - 8, bottom + 8), str(x_value), fill=INK, font=font)

    for item in series:
        values = list(item.get("values", []))
        color = item.get("color", PRIMARY)
        points = []
        for x_index, value in enumerate(values):
            x = left + (x_index / max(len(x_values) - 1, 1)) * plot_w
            y = top + plot_h - ((value - y_bottom) / max(y_top - y_bottom, 1e-9)) * plot_h
            points.append((x, y))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        errors = item.get("errors")
        if errors:
            for x_index, error in enumerate(errors):
                if error is None:
                    continue
                x = left + (x_index / max(len(x_values) - 1, 1)) * plot_w
                y = top + plot_h - ((values[x_index] - y_bottom) / max(y_top - y_bottom, 1e-9)) * plot_h
                draw.line((x, y - error * 80, x, y + error * 80), fill=color, width=2)

    legend_y = 40
    for item in series:
        draw.line((right - 240, legend_y, right - 210, legend_y), fill=item.get("color", PRIMARY), width=3)
        draw.text((right - 200, legend_y - 8), item["label"], fill=INK, font=font)
        legend_y += 25

    img.save(output_path, format="PNG")
    return output_path
