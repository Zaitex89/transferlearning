# Transfer Learning Flower Classifier

An image classifier built by transfer learning: instead of training a CNN from
scratch, we reuse MobileNetV2 already trained on ImageNet (1.4M images, 1000
classes), freeze what it learned, and train only a small new head on our own data.

- Task: given a photo, predict which of 5 flower species it shows (single-label, 5-way classification)
- Dataset: tf_flowers — 3,670 Creative Commons Flickr photos
  - daisy 633 · dandelion 898 · roses 641 · sunflowers 699 · tulips 799
  - labels come from the folder name; class indices are assigned alphabetically
  - real-world photos: mixed resolutions (~160–500 px), varied lighting, backgrounds
    and framing, and some genuinely loose labels (garden scenes tagged "tulips"),
    which is why ~95% is about the practical ceiling — see `outputs/dataset_samples.png`
- Backbone: EfficientNetV2B0, ImageNet weights, 270 layers, 5.9M params
- Split: 70% train (2,569) / 15% validation (557) / 15% test (544)
- Result: 95.59% test accuracy** (520/544 held-out images)
- Hardware: trains on CPU in ~20 minutes, no GPU needed

## Run it

```bash
python train.py          # trains both stages, writes everything to outputs/
python predict.py some_photo.jpg
```

The dataset (218 MB) downloads automatically on the first run and is cached.

## How the transfer learning works

### Stage 1 — feature extraction (backbone frozen)

```python
base = keras.applications.MobileNetV2(include_top=False, weights="imagenet")
base.trainable = False          # <- the freeze
```

The whole convolutional backbone is locked. Its job is to turn an image into a
1280-number "fingerprint" describing edges, textures and shapes — features that
are just as useful for flowers as for the ImageNet objects it was trained on.
The only thing that learns is the new head:

```
GlobalAveragePooling2D -> Dropout(0.3) -> Dense(5, softmax)
```

That's 6,405 trainable parameters out of 2.3 million. Training is fast and
can't overfit much, because there is almost nothing to overfit with. A high
learning rate (1e-3) is fine here — we're only fitting a linear classifier.

### Stage 2 — fine-tuning (top layers unfrozen)

```python
base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False     # bottom 100 layers stay frozen
```

Early convolutional layers detect generic things (edges, colour blobs) — those
transfer perfectly and should stay frozen. The late layers detect
ImageNet-specific concepts, so we unfreeze layers 100–154 and let them
re-specialise on flowers. Two details that matter a lot here:

1. -Learning rate drops to 1e-5 (100x smaller). With a large learning rate
   the first noisy gradients from the randomly-initialised head would destroy
   the pretrained filters — the thing we came for.

2. -BatchNorm layers stay frozen. They carry ImageNet mean/variance
   statistics; recomputing them on 2,500 flower photos makes results worse.
   This is the single most common transfer learning bug.

### Keeping accuracy honest

- Augmentation (flip / rotate / zoom / contrast) is applied as Keras layers
  inside the model, so it runs only during training, never at inference.
- Preprocessing is baked in too (`Rescaling` to [-1, 1], what MobileNetV2
  expects), so `predict.py` can hand the model a raw image and can't get it wrong.
- Three-way split. Validation drives early stopping and the LR schedule, so
  it is no longer an unbiased estimate. The test split is touched exactly once,
  at the very end — that number is the one worth reporting.
- Callbacks: `EarlyStopping` (restores best weights), `ModelCheckpoint`
  (keeps the best val-accuracy model), `ReduceLROnPlateau`.

## Outputs

| File | What it is |
|---|---|
| `outputs/dataset_samples.png` | 5 example images per class |
| `outputs/flower_model.keras` | trained model, preprocessing included |
| `outputs/training_curves.png` | accuracy/loss, with the fine-tune point marked |
| `outputs/confusion_matrix.png` | test-set confusion matrix |
| `outputs/metrics.json` | test accuracy + per-class precision/recall |
| `outputs/train_log.txt` | full training log |

## Results

Final: 95.59% test accuracy (520/544), 12 head epochs + 22 fine-tune epochs
(early-stopped from 28), ~20 min on a 12-core CPU.

| Class | Precision | Recall | Support |
|---|---|---|---|
| daisy | 0.967 | 0.978 | 90 |
| dandelion | 0.976 | 0.984 | 126 |
| roses | 0.903 | 0.949 | 98 |
| sunflowers | 0.974 | 0.949 | 118 |
| tulips | 0.954 | 0.920 | 112 |

### What each change was worth

| Configuration | Test accuracy | Errors |
|---|---|---|
| MobileNetV2, 12+10 epochs | 92.10% | 43 / 544 |
| EfficientNetV2B0 + label smoothing, 12+12 | 95.22% | 26 / 544 |
| EfficientNetV2B0 + label smoothing, 12+22 | 95.59% | 24 / 544 |

Archived runs are in `baseline_mobilenetv2/` and `baseline_effnet_12ep/`.

The big win was the backbone swap — 40% fewer errors, and EfficientNetV2B0 is
faster than MobileNetV2 on CPU (130 vs 123 img/s) despite being 2.5x larger,
because it's better optimised. In transfer learning the quality of the pretrained
features usually matters more than anything done to the head.

### Remaining errors

roses↔tulips is ~40% of all mistakes. Both are dense, saturated, multi-petal
flowers, often photographed in mixed beds. Tulips shows high precision (0.954) but
lower recall (0.920) — the model rarely says "tulip" wrongly, it just misses them,
biasing toward "roses" on ambiguous images. Some of this is unfixable: the dataset
contains genuinely loose labels (wide garden scenes filed under `tulips`), so ~96%
is close to the practical ceiling.

## Useful flags

```bash
python train.py --no-finetune              # stage 1 only, to compare
python train.py --unfreeze_from 120        # unfreeze fewer layers
python train.py --data_dir my_images       # your own data: my_images/<class>/*.jpg
python train.py --img_size 160             # ~2x faster, slightly lower accuracy
```

## Using your own images

Lay them out one folder per class and pass `--data_dir`. Nothing else changes —
the class count, names and output layer all adapt automatically:

```
my_images/
  cats/  img1.jpg ...
  dogs/  img1.jpg ...
```
