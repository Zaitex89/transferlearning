"""
Classify images with the trained model.

    python predict.py                           # defaults to sample_images/
    python predict.py path/to/photo.jpg
    python predict.py path/to/folder            # every image in it, recursively
"""

import json
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TF's startup noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import numpy as np

HERE = pathlib.Path(__file__).parent
OUT_DIR = HERE / "outputs"
DEFAULT_TARGET = HERE / "sample_images"  # used when no path is given
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def main():
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET

    if not target.exists():
        sys.exit(f"Path does not exist: {target}")

    meta = json.loads((OUT_DIR / "metrics.json").read_text())
    class_names, img_size = meta["class_names"], meta["img_size"]
    model = keras.models.load_model(OUT_DIR / "flower_model.keras")

    # rglob so a folder of class subfolders works too, not just a flat folder
    paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in EXTS) if target.is_dir() else [target]
    if not paths:
        sys.exit(f"No images found in {target} (looked for {', '.join(sorted(EXTS))})")

    batch = np.stack([
        keras.utils.img_to_array(keras.utils.load_img(p, target_size=(img_size, img_size)))
        for p in paths
    ])
    probs = model.predict(batch, verbose=0)

    for path, prob in zip(paths, probs):
        order = prob.argsort()[::-1][:3]
        top = "  ".join(f"{class_names[i]} {prob[i]:.1%}" for i in order)
        print(f"{path.name:<40} -> {class_names[order[0]]:<12} | {top}")


if __name__ == "__main__":
    main()
