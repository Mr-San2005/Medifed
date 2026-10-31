"""Create a deterministic four-class Grad-CAM demonstration image."""

from __future__ import annotations

import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medifed.config import CLASS_NAMES, HOSPITAL_RAW_DIRS, PROCESSED_DIR, RESULTS_DIR
from medifed.inference import predict


def demo_rows() -> list[dict]:
    """Select the first validation example for each class across hospitals."""
    selected: dict[str, dict] = {}
    for hospital_id in (1, 2, 3):
        manifest = PROCESSED_DIR / f"hospital{hospital_id}_manifest.csv"
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "val" and row["class_name"] not in selected:
                    selected[row["class_name"]] = {
                        "true_class": row["class_name"],
                        "hospital": hospital_id,
                        "path": HOSPITAL_RAW_DIRS[hospital_id] / row["relative_path"],
                    }
    return [selected[name] for name in CLASS_NAMES if name in selected]


def build_demo_image(rows: list[dict], variant: str) -> tuple[Image.Image, list[dict]]:
    """Run inference and compose original/overlay/result rows into a PNG."""
    font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=12)
    row_height = 248
    width = 960
    canvas = Image.new("RGB", (width, 112 + row_height * len(rows)), "#F5F7F8")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 24), "MediFed | Four-class Grad-CAM demonstration", fill="#1B2A33", font=font)
    draw.text((28, 56), f"Model variant: {variant}   |   Research prototype, not for clinical use", fill="#5B6B75", font=small_font)
    output_rows = []
    for index, row in enumerate(rows):
        if "path" in row:
            image_bytes = row["path"].read_bytes()
            true_class = row["true_class"]
        else:
            image_bytes = row["image_bytes"]
            true_class = "UNLABELLED"
        result = predict(variant, image_bytes)
        original = Image.open(BytesIO(result["original_png"])).convert("RGB")
        heatmap = Image.open(BytesIO(result["heatmap_png"])).convert("RGB")
        y = 94 + index * row_height
        draw.line((24, y, width - 24, y), fill="#D8E0E4", width=1)
        canvas.paste(original.resize((200, 200)), (28, y + 20))
        canvas.paste(heatmap.resize((200, 200)), (248, y + 20))
        top_three = ", ".join(f"{item['class']} {item['probability']:.1%}" for item in result["ranked"][:3])
        draw.text((476, y + 28), f"True class: {true_class}", fill="#1B2A33", font=font)
        if "hospital" in row:
            draw.text((476, y + 50), f"Input hospital: {row['hospital']}", fill="#5B6B75", font=small_font)
        draw.text((476, y + 68), f"Predicted: {result['top_class']}", fill="#0F6B6E", font=font)
        draw.text((476, y + 100), "Top 3 scores:", fill="#5B6B75", font=small_font)
        draw.text((476, y + 122), top_three, fill="#1B2A33", font=small_font)
        output_rows.append({"true_class": true_class, "hospital": row.get("hospital", ""), "prediction": result["top_class"], "top_three": top_three})
    draw.line((24, canvas.height - 18, width - 24, canvas.height - 18), fill="#D8E0E4", width=1)
    return canvas, output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a four-class MediFed Grad-CAM demo.")
    parser.add_argument("--variant", default="fedbn_hospital3")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    rows = demo_rows() if args.image is None else [{"image_bytes": args.image.read_bytes()}]
    image, predictions = build_demo_image(rows, args.variant)
    output = RESULTS_DIR / f"gradcam_demo_{args.variant}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print("true_class\thospital\tpredicted\ttop_three")
    for row in predictions:
        print(f"{row['true_class']}\t{row['hospital']}\t{row['prediction']}\t{row['top_three']}")
    print(f"Demo image: {output}")


if __name__ == "__main__":
    main()