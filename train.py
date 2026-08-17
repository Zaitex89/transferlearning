"""
Transfer learning image classifier.

Stage 1 (feature extraction): the ImageNet-pretrained MobileNetV2 backbone is
frozen and only a new classification head is trained.
Stage 2 (fine-tuning): the top block of the backbone is unfrozen and trained
with a very small learning rate so the pretrained filters are nudged, not wrecked.

Default dataset is tf_flowers (5 classes, 3670 photos). Point --data_dir at any
folder laid out as  data_dir/<class_name>/<image>.jpg  to use your own images.

    python train.py                     # full run, default settings
    python train.py --no-finetune       # stage 1 only
    python train.py --data_dir my_data  # your own images
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
OUT_DIR = pathlib.Path(__file__).parent / "outputs"
MODEL_PATH = OUT_DIR / "flower_model.keras"

# name -> (constructor, how the backbone wants its pixels)
#   "builtin" = the model rescales internally and wants raw 0-255
#   "signed"  = we must scale to [-1, 1] ourselves
BACKBONES = {
    "efficientnetv2b0": (keras.applications.EfficientNetV2B0, "builtin"),
    "mobilenetv2": (keras.applications.MobileNetV2, "signed"),
    "resnet50v2": (keras.applications.ResNet50V2, "signed"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None, help="folder with one subfolder per class (default: download tf_flowers)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_head", type=int, default=12, help="stage 1 epochs (frozen backbone)")
    p.add_argument("--epochs_finetune", type=int, default=12, help="stage 2 epochs (top block unfrozen)")
    p.add_argument("--backbone", default="efficientnetv2b0", choices=sorted(BACKBONES))
    p.add_argument("--unfreeze_frac", type=float, default=0.35,
                   help="fraction of the backbone (from the top) to unfreeze in stage 2")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="softens targets; helps when some labels are noisy")
    p.add_argument("--no-finetune", dest="finetune", action="store_false", help="skip stage 2")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


class Tee:
    """Mirror the console into outputs/train_log.txt, so the log is a run artifact
    instead of something you have to remember to redirect."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "w", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        # Keras asks this to decide between progress bars and plain lines.
        return self.stream.isatty()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


def get_data_dir(user_dir):
    """Return the image root, downloading tf_flowers on first run."""
    if user_dir:
        return pathlib.Path(user_dir)
    archive = keras.utils.get_file(origin=DATA_URL, extract=True, cache_subdir="datasets")
    # Keras 3 extracts into "<archive>_extracted/flower_photos"; older layouts drop it alongside.
    for candidate in (
        pathlib.Path(archive) / "flower_photos",
        pathlib.Path(archive).with_suffix("").with_suffix("") / "flower_photos",
        pathlib.Path(archive).parent / "flower_photos",
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not locate flower_photos under {archive}")


def build_datasets(data_dir, img_size, batch_size, seed):
    """70% train / 15% validation / 15% test, cached in RAM as uint8."""
    train_ds, holdout_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="both",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",  # one-hot, so the loss can apply label smoothing
    )
    class_names = train_ds.class_names
    val_ds = holdout_ds.shard(2, 0)  # even batches -> validation (drives early stopping)
    test_ds = holdout_ds.shard(2, 1)  # odd batches  -> test (touched once, at the end)

    def prep(ds):
        # uint8 keeps the RAM cache ~4x smaller; the model rescales to float itself.
        ds = ds.map(lambda x, y: (tf.cast(x, tf.uint8), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    return prep(train_ds), prep(val_ds), prep(test_ds), class_names


def build_model(img_size, num_classes, backbone_name):
    augment = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.15),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    constructor, pixel_mode = BACKBONES[backbone_name]
    base = constructor(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False  # freeze the backbone (also puts its BatchNorm in inference mode)

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = augment(inputs)  # only active during training
    if pixel_mode == "signed":
        # MobileNetV2/ResNetV2 expect [-1, 1]. EfficientNetV2 rescales internally, so
        # adding our own layer there would double-scale the input and tank accuracy.
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name=f"{backbone_name}_transfer"), base


def compile_model(model, lr, label_smoothing):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def callbacks_for(patience):
    return [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]


def plot_dataset_samples(ds, class_names, per_class=5):
    """Grid of raw training images, one row per class — what the model actually sees
    before augmentation. Drawn before training so it survives a crashed run."""
    examples = {i: [] for i in range(len(class_names))}
    for images, labels in ds:  # uint8 batches, one-hot labels
        for image, label in zip(images.numpy(), labels.numpy().argmax(axis=1)):
            if len(examples[label]) < per_class:
                examples[label].append(image)
        if all(len(v) >= per_class for v in examples.values()):
            break

    fig, axes = plt.subplots(
        len(class_names), per_class, squeeze=False,
        figsize=(1.6 * per_class, 1.75 * len(class_names)),
    )
    for row, name in enumerate(class_names):
        for col in range(per_class):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(examples[row]):
                ax.imshow(examples[row][col])
        axes[row][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(f"{per_class} examples per class (before augmentation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_samples.png", dpi=120)
    plt.close(fig)


def plot_history(hist_head, hist_ft, split_at):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        train = hist_head.history[key] + (hist_ft.history[key] if hist_ft else [])
        val = hist_head.history["val_" + key] + (hist_ft.history["val_" + key] if hist_ft else [])
        ax.plot(train, label="train")
        ax.plot(val, label="validation")
        if hist_ft:
            ax.axvline(split_at - 0.5, color="gray", ls="--", lw=1)
            ax.text(split_at - 0.4, ax.get_ylim()[0], " fine-tune", fontsize=8, color="gray", va="bottom")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=120)
    plt.close(fig)


def plot_confusion(cm, class_names):
    fig, ax = plt.subplots(figsize=(1.3 * len(class_names) + 2, 1.1 * len(class_names) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def evaluate(model, test_ds, class_names):
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).argmax(axis=1)
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    acc = float((y_true == y_pred).mean())

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plot_confusion(cm, class_names)

    print(f"\nTest accuracy: {acc:.4f}  ({int((y_true == y_pred).sum())}/{len(y_true)} images)\n")
    print(f"{'class':<14}{'precision':>10}{'recall':>9}{'support':>9}")
    per_class = {}
    for i, name in enumerate(class_names):
        prec = cm[:, i].sum() and cm[i, i] / cm[:, i].sum()
        rec = cm[i, :].sum() and cm[i, i] / cm[i, :].sum()
        per_class[name] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                           "support": int(cm[i, :].sum())}
        print(f"{name:<14}{prec:>10.3f}{rec:>9.3f}{cm[i, :].sum():>9}")
    return acc, per_class


def run(args):
    data_dir = get_data_dir(args.data_dir)
    print(f"Dataset: {data_dir}")
    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir, args.img_size, args.batch_size, args.seed
    )
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    plot_dataset_samples(train_ds, class_names)

    model, base = build_model(args.img_size, len(class_names), args.backbone)
    trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"Backbone: {args.backbone} - {len(base.layers)} layers, {base.count_params():,} params (frozen)")
    print(f"Trainable now: {trainable:,} params (head only)\n")

    print("=" * 60, "\nSTAGE 1 - frozen backbone, training the head\n", "=" * 60, sep="")
    compile_model(model, lr=1e-3, label_smoothing=args.label_smoothing)
    hist_head = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                          callbacks=callbacks_for(patience=4), verbose=2)

    hist_ft = None
    if args.finetune:
        cut = int(len(base.layers) * (1 - args.unfreeze_frac))
        print("\n" + "=" * 60,
              f"\nSTAGE 2 - fine-tuning backbone layers {cut}-{len(base.layers)}\n", "=" * 60, sep="")
        base.trainable = True
        for layer in base.layers[:cut]:
            layer.trainable = False
        for layer in base.layers:
            # BatchNorm stats come from ImageNet; updating them on a small dataset hurts.
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        # 100x smaller LR: nudge the filters, don't wreck them
        compile_model(model, lr=1e-5, label_smoothing=args.label_smoothing)
        trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
        print(f"Now trainable: {trainable:,} params\n")
        hist_ft = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                            callbacks=callbacks_for(patience=4), verbose=2)

    plot_history(hist_head, hist_ft, split_at=len(hist_head.history["accuracy"]))

    print("\n" + "=" * 60, "\nEVALUATION on the held-out test split\n", "=" * 60, sep="")
    acc, per_class = evaluate(model, test_ds, class_names)

    model.save(MODEL_PATH)
    (OUT_DIR / "metrics.json").write_text(json.dumps({
        "test_accuracy": round(acc, 4),
        "backbone": args.backbone,
        "class_names": class_names,
        "per_class": per_class,
        "img_size": args.img_size,
        "fine_tuned": args.finetune,
        "label_smoothing": args.label_smoothing,
    }, indent=2))
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved plots + metrics + log -> {OUT_DIR}")


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    OUT_DIR.mkdir(exist_ok=True)

    # Tee before anything prints, and keep the log even if the run raises.
    # stderr goes to the same file so Python-level warnings and tracebacks land in
    # it too (TF's C++ logs bypass Python and stay console-only).
    stdout, stderr = sys.stdout, sys.stderr
    tee = Tee(stdout, OUT_DIR / "train_log.txt")
    sys.stdout = sys.stderr = tee
    try:
        run(args)
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        tee.file.close()


if __name__ == "__main__":
    main()
