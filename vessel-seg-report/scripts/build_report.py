from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "figures" / "wandb" / "training"
PRED_DIR = ROOT / "figures" / "wandb" / "prediction"
REPORT_IMG_DIR = ROOT / "figures" / "report_images"
PLOT_DIR = ROOT / "figures" / "plots"
LOG_DIR = ROOT / "source_logs"

FOLD_INFO = {
    1: {
        "tiff": "13_20250724_Plate1_C3_D7_MAX_Crop.tif",
        "epochs": [10, 20, 30, 40, 50, 60, 70],
        "best_epoch": 48,
        "val_loss": 0.0568,
        "selected": True,
    },
    2: {
        "tiff": "42_20250807_Plate1_C2_D21_MAX_Crop.tif",
        "epochs": [10, 20, 30, 40, 50, 60],
        "best_epoch": 33,
        "val_loss": 0.0713,
        "selected": False,
    },
    3: {
        "tiff": "2_20260418_EC_Plate1_A2_D7_MAX_Crop.tif",
        "epochs": [10, 20, 30, 40, 50, 60],
        "best_epoch": 37,
        "val_loss": 0.0627,
        "selected": False,
    },
    4: {
        "tiff": "7_20250724_Plate1_B2_D7_MAX_Crop.tif",
        "epochs": [10, 20, 30, 40, 50],
        "best_epoch": 23,
        "val_loss": 0.0624,
        "selected": False,
    },
    5: {
        "tiff": "18_20250731_Plate1_A3_D14_MAX_Crop.tif",
        "epochs": [10, 20, 30, 40, 50],
        "best_epoch": 22,
        "val_loss": 0.0721,
        "selected": False,
    },
}

PRED_INFO = [
    (0, "16_20250731_Plate1_A1_D14_MAX_Crop.tif", "train/validation pool", False),
    (1, "17_20260418_EC_Plate1_A2_D14_MAX_Crop.tif", "test", True),
    (2, "20_20260418_EC_Plate2_A2_D14_MAX_Crop.tif", "train/validation pool", False),
    (3, "32_20250807_Plate1_A2_D21_MAX_Crop.tif", "test", True),
    (4, "43_20250807_Plate1_C3_D21_MAX_Crop.tif", "test", True),
    (5, "12_20250724_Plate1_C2_D7_MAX_Crop.tif", "test", True),
    (6, "1_20250724_Plate1_A1_D7_MAX_Crop.tif", "test", True),
]

TEST_DICE = {
    "12_20250724_Plate1_C2_D7_MAX_Crop.tif": 0.9154,
    "32_20250807_Plate1_A2_D21_MAX_Crop.tif": 0.9110,
    "1_20250724_Plate1_A1_D7_MAX_Crop.tif": 0.9036,
    "43_20250807_Plate1_C3_D21_MAX_Crop.tif": 0.9141,
    "17_20260418_EC_Plate1_A2_D14_MAX_Crop.tif": 0.9507,
}

TRAIN_PANELS = [
    "input_image",
    "ch_grad",
    "ch_sharp",
    "gt_mask",
    "pred_prob",
    "pred_binary",
    "attn1_overlay",
    "attn2_overlay",
    "attn3_overlay",
]

FINAL_PANELS = [
    ("originals", "Original image"),
    ("ch_grad", "Gradient magnitude"),
    ("ch_sharp", "Sharpness map"),
    ("gt", "Ground truth"),
    ("prediction", "Prediction"),
    ("tp", "True positive"),
    ("fp", "False positive"),
    ("fn", "False negative"),
    ("combined", "TP/FP/FN overlay"),
]


def tex_file(name: str) -> str:
    return rf"\nolinkurl{{{name}}}"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resize_for_report(src: Path, dst: Path, max_width: int) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        with Image.open(dst) as existing:
            if existing.width <= max_width:
                return dst
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > max_width:
            ratio = max_width / im.width
            new_size = (max_width, max(1, int(im.height * ratio)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        im.save(dst, optimize=True)
    return dst


def parse_media(folder: Path) -> dict[str, dict[int, Path]]:
    pattern = re.compile(r"(?P<panel>.+)_(?P<step>\d+)_[^.]+\.png$")
    grouped: dict[str, dict[int, Path]] = {}
    for item in folder.glob("*.png"):
        match = pattern.match(item.name)
        if not match:
            continue
        grouped.setdefault(match.group("panel"), {})[int(match.group("step"))] = item
    return grouped


def training_lookup() -> dict[int, dict[str, dict[int, Path]]]:
    lookup: dict[int, dict[str, dict[int, Path]]] = {}
    for fold, info in FOLD_INFO.items():
        grouped = parse_media(TRAIN_DIR / f"fold{fold}")
        fold_lookup: dict[str, dict[int, Path]] = {}
        for panel in TRAIN_PANELS:
            items = sorted(grouped.get(panel, {}).items())
            if len(items) != len(info["epochs"]):
                raise RuntimeError(f"fold {fold} panel {panel} has {len(items)} files")
            fold_lookup[panel] = {
                epoch: path for epoch, (_, path) in zip(info["epochs"], items)
            }
        lookup[fold] = fold_lookup
    return lookup


def prediction_lookup() -> dict[int, dict[str, Path]]:
    grouped = parse_media(PRED_DIR)
    lookup: dict[int, dict[str, Path]] = {}
    for index, _, _, _ in PRED_INFO:
        lookup[index] = {}
        for panel, _ in FINAL_PANELS:
            if index not in grouped.get(panel, {}):
                raise RuntimeError(f"prediction index {index} missing {panel}")
            lookup[index][panel] = grouped[panel][index]
    return lookup


def write_training_curves() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\| train (?P<train_loss>[0-9.]+) "
        r"dice=(?P<train_dice>[0-9.]+).*?\| val (?P<val_loss>[0-9.]+) "
        r"dice=(?P<val_dice>[0-9.]+)"
    )
    fold = 0
    curves: dict[int, list[dict[str, str]]] = {}
    for line in (LOG_DIR / "train_6670898.out").read_text().splitlines():
        fold_match = re.search(r"Fold (?P<fold>\d+)/5", line)
        if fold_match:
            fold = int(fold_match.group("fold"))
            curves.setdefault(fold, [])
            continue
        match = pattern.search(line)
        if match and fold:
            curves[fold].append(match.groupdict())

    for fold_id, rows in curves.items():
        with (PLOT_DIR / f"fold{fold_id}_curves.dat").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["epoch", "train_loss", "val_loss", "train_dice", "val_dice"],
                delimiter=" ",
            )
            writer.writeheader()
            writer.writerows(rows)


def write_manifests(train_paths: dict, pred_paths: dict) -> None:
    train_rows = []
    for fold, panels in train_paths.items():
        info = FOLD_INFO[fold]
        for panel in TRAIN_PANELS:
            for epoch in info["epochs"]:
                src = panels[panel][epoch]
                thumb = resize_for_report(
                    src,
                    REPORT_IMG_DIR / "training" / src.relative_to(TRAIN_DIR),
                    max_width=420,
                )
                train_rows.append(
                    {
                        "figure_group": f"fold{fold}_progress",
                        "wandb_source_file": rel(src),
                        "report_file": rel(thumb),
                        "original_tiff": info["tiff"],
                        "split_type": "validation",
                        "fold": fold,
                        "epoch": epoch,
                        "panel_type": panel,
                        "description": f"Fold {fold} validation {panel} at epoch {epoch}",
                    }
                )

    with (ROOT / "figures" / "manifest_training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=train_rows[0].keys())
        writer.writeheader()
        writer.writerows(train_rows)

    pred_rows = []
    info_by_index = {item[0]: item for item in PRED_INFO}
    for index, panels in pred_paths.items():
        _, tiff, split_type, held_out = info_by_index[index]
        for panel, label in FINAL_PANELS:
            src = panels[panel]
            thumb = resize_for_report(
                src,
                REPORT_IMG_DIR / "prediction" / src.relative_to(PRED_DIR),
                max_width=2400,
            )
            pred_rows.append(
                {
                    "figure_group": f"final_prediction_{index}",
                    "wandb_source_file": rel(src),
                    "report_file": rel(thumb),
                    "original_tiff": tiff,
                    "split_type": split_type,
                    "held_out_from_training": "yes" if held_out else "no",
                    "fold": "",
                    "epoch": "",
                    "panel_type": panel,
                    "description": f"{label} for {tiff}",
                }
            )

    with (ROOT / "figures" / "manifest_prediction.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pred_rows[0].keys())
        writer.writeheader()
        writer.writerows(pred_rows)


def image_path_from_manifest(
    train_paths: dict[int, dict[str, dict[int, Path]]], fold: int, panel: str, epoch: int
) -> str:
    src = train_paths[fold][panel][epoch]
    thumb = REPORT_IMG_DIR / "training" / src.relative_to(TRAIN_DIR)
    return rel(thumb)


def pred_image_path(
    pred_paths: dict[int, dict[str, Path]], index: int, panel: str
) -> str:
    src = pred_paths[index][panel]
    thumb = REPORT_IMG_DIR / "prediction" / src.relative_to(PRED_DIR)
    return rel(thumb)


def training_progress_figure(train_paths: dict, fold: int) -> str:
    info = FOLD_INFO[fold]
    epochs = info["epochs"]
    tiff = info["tiff"]
    ncols = len(epochs)
    colspec = "l" + ("c" * ncols)
    progress_height = "0.088\\linewidth"
    reference_height = "0.105\\linewidth"
    lines = [
        r"\begin{landscape}",
        r"\begin{figure}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        rf"\textbf{{Fold {fold} validation image: {tex_file(tiff)}}}\\[0.4em]",
        r"\begin{tabular}{lcccc}",
        r" & Input & Gradient magnitude & Sharpness map & Ground truth \\",
        rf"Reference & \includegraphics[height={reference_height}]{{{image_path_from_manifest(train_paths, fold, 'input_image', epochs[0])}}}"
        rf" & \includegraphics[height={reference_height}]{{{image_path_from_manifest(train_paths, fold, 'ch_grad', epochs[0])}}}"
        rf" & \includegraphics[height={reference_height}]{{{image_path_from_manifest(train_paths, fold, 'ch_sharp', epochs[0])}}}"
        rf" & \includegraphics[height={reference_height}]{{{image_path_from_manifest(train_paths, fold, 'gt_mask', epochs[0])}}} \\",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        rf"\begin{{tabular}}{{{colspec}}}",
        " & " + " & ".join([f"Epoch {epoch}" for epoch in epochs]) + r" \\",
    ]
    rows = [
        ("Probability", "pred_prob"),
        ("Binary mask", "pred_binary"),
        ("Attention gate 1", "attn1_overlay"),
        ("Attention gate 2", "attn2_overlay"),
        ("Attention gate 3", "attn3_overlay"),
    ]
    for label, panel in rows:
        cells = [
            rf"\includegraphics[height={progress_height}]{{{image_path_from_manifest(train_paths, fold, panel, epoch)}}}"
            for epoch in epochs
        ]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\end{tabular}",
            rf"\caption{{Step-by-step training progress for validation image {tex_file(tiff)} in fold {fold}. "
            rf"The split type is validation and the epochs shown are {', '.join(map(str, epochs))}. "
            "The top reference row shows the input image, engineered gradient channel, sharpness channel, and ground-truth vessel mask. "
            "The epoch rows show probability maps, thresholded binary masks, and attention overlays from gates 1--3. "
            "Hotter attention colors indicate regions receiving stronger gating weight during decoding.}",
            rf"\label{{fig:fold{fold}-progress}}",
            r"\end{figure}",
            r"\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def attention_only_figure(train_paths: dict, fold: int) -> str:
    info = FOLD_INFO[fold]
    epochs = info["epochs"]
    tiff = info["tiff"]
    ncols = len(epochs)
    colspec = "l" + ("c" * ncols)
    attention_height = "0.14\\linewidth"
    lines = [
        r"\begin{landscape}",
        r"\begin{figure}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\begin{{tabular}}{{{colspec}}}",
        " & " + " & ".join([f"Epoch {epoch}" for epoch in epochs]) + r" \\",
    ]
    for label, panel in [
        ("Gate 1", "attn1_overlay"),
        ("Gate 2", "attn2_overlay"),
        ("Gate 3", "attn3_overlay"),
    ]:
        cells = [
            rf"\includegraphics[height={attention_height}]{{{image_path_from_manifest(train_paths, fold, panel, epoch)}}}"
            for epoch in epochs
        ]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\end{tabular}",
            rf"\caption{{Attention-only view for validation image {tex_file(tiff)} in fold {fold}. "
            rf"The split type is validation and columns are epochs {', '.join(map(str, epochs))}. "
            "Rows correspond to the three attention gates in the decoder, allowing coarse-to-fine changes in feature selection to be inspected without the probability-mask rows.}",
            rf"\label{{fig:fold{fold}-attention}}",
            r"\end{figure}",
            r"\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def final_prediction_figure(pred_paths: dict, index: int, tiff: str, split: str, held_out: bool) -> str:
    held_text = "held out from training" if held_out else "from the train/validation pool"
    split_phrase = "held-out test image" if held_out else "train/validation pool image"
    dice_sentence = ""
    if tiff in TEST_DICE:
        dice_sentence = f" The stitched-image Dice score for this held-out test image was {TEST_DICE[tiff]:.4f}."
    lines = [
        r"\begin{landscape}",
        r"\begin{figure}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{p{0.16\linewidth}p{0.78\linewidth}}",
    ]
    for panel, label in FINAL_PANELS:
        lines.append(
            rf"\textbf{{{label}}} & \includegraphics[width=0.78\linewidth]{{{pred_image_path(pred_paths, index, panel)}}} \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            rf"\caption{{Final stacked comparison for {split_phrase} {tex_file(tiff)}. "
            rf"The split type is {split}; this image was {held_text}. "
            "Panels are ordered from original image, gradient magnitude, sharpness map, ground truth, prediction, true positive, false positive, false negative, to the combined TP/FP/FN overlay. "
            "Green denotes true-positive vessel pixels, red denotes false positives, and yellow denotes false negatives."
            rf"{dice_sentence}}}",
            rf"\label{{fig:final-{index}}}",
            r"\end{figure}",
            r"\end{landscape}",
            "",
            rf"\noindent {tex_file(tiff)} is a {split_phrase}. The stacked comparison shows whether the model preserved continuous vessel structure and where the remaining errors concentrate. "
            "Successful regions appear as long green vessel traces in the true-positive and combined panels. "
            "Residual red regions indicate small non-vessel structures selected by the model, while yellow regions mark faint or fragmented vessel segments that remain difficult under low contrast and uneven focus.",
            "",
        ]
    )
    return "\n".join(lines)


def detailed_architecture_figure() -> str:
    return r"""
\begin{landscape}
\begin{figure}[p]
\centering
\resizebox{0.99\linewidth}{!}{%
\begin{tikzpicture}[
font=\sffamily\scriptsize,
imagecard/.style={draw=white, line width=1.2pt, fill=white, rounded corners=2pt, inner sep=1.2pt, blur shadow={shadow blur steps=6, shadow opacity=0.22}},
mainarrow/.style={-{Latex[length=2.8mm,width=1.9mm]}, line width=1.05pt, draw=black!62},
skiparrow/.style={-{Latex[length=2.4mm,width=1.7mm]}, line width=0.92pt, draw=MidnightBlue!78},
gatearrow/.style={-{Latex[length=2.2mm,width=1.55mm]}, line width=0.88pt, draw=BurntOrange!82!black},
gate/.style={circle, minimum size=0.70cm, inner sep=1pt, draw=BurntOrange!72!black, fill=Orange!35, line width=0.72pt, font=\sffamily\tiny\bfseries, blur shadow={shadow blur steps=4, shadow opacity=0.12}},
head/.style={rounded corners=3pt, draw=ForestGreen!55!black, fill=ForestGreen!12, line width=0.75pt, minimum width=0.78cm, minimum height=0.72cm, align=center, blur shadow={shadow blur steps=4, shadow opacity=0.12}},
label/.style={font=\sffamily\scriptsize\bfseries, text=black!68, align=center},
tinylabel/.style={font=\sffamily\tiny, text=black!58, align=center}
]

\newcommand{\FeatureStack}[7]{%
  \begin{scope}[shift={#2}]
    \foreach \i/\shade in {0/50,1/40,2/31,3/23}{%
      \filldraw[draw=#5!72!black, fill=#5!\shade, line width=0.60pt, line join=round]
        ({0.13*\i},{0.09*\i}) -- ++(#3,0.26) -- ++(0,#4) -- ++(-#3,-0.26) -- cycle;
    }
    \coordinate (#1west) at (-0.03,{#4/2+0.12});
    \coordinate (#1east) at ({#3+0.50},{#4/2+0.38});
    \coordinate (#1north) at ({#3/2+0.24},{#4+0.33});
    \coordinate (#1south) at ({#3/2+0.24},-0.04);
    \ifstrempty{#6}{}{\node[tinylabel] at ({#3/2+0.24},-0.42) {#6};}
  \end{scope}
}

\node[font=\sffamily\small\bfseries, align=center] at (0,3.72) {Image-aware Attention U-Net};

% Upright input image stack
\node[imagecard] (img1) at (-6.80,1.86)
  {\includegraphics[width=1.30cm,height=1.62cm]{figures/report_images/prediction/originals_0_b01aa3ecab637aa430ca.png}};
\node[imagecard] (img2) at (-6.58,2.02)
  {\includegraphics[width=1.30cm,height=1.62cm]{figures/report_images/prediction/ch_grad_0_74cf9a898842fdc974eb.png}};
\node[imagecard] (img3) at (-6.36,2.18)
  {\includegraphics[width=1.30cm,height=1.62cm]{figures/report_images/prediction/ch_sharp_0_19a8b365298f5e350ecf.png}};
\node[tinylabel] at (-6.58,0.78) {3-channel input};
\node[tinylabel] at (-6.58,0.50) {image | gradient | sharpness};

% Encoder path
\FeatureStack{e0}{(-4.80,1.72)}{0.70}{1.28}{MidnightBlue}{}{}
\FeatureStack{e1}{(-4.10,0.50)}{0.58}{1.12}{MidnightBlue}{}{}
\FeatureStack{e2}{(-3.38,-0.70)}{0.48}{0.96}{MidnightBlue}{}{}
\FeatureStack{e3}{(-2.70,-1.90)}{0.38}{0.82}{MidnightBlue}{}{}
\FeatureStack{bn}{(-0.45,-2.86)}{0.42}{0.70}{Purple}{bottleneck}{}

% Decoder path
\FeatureStack{d3}{(2.10,-1.90)}{0.38}{0.82}{BurntOrange}{}{}
\FeatureStack{d2}{(2.78,-0.70)}{0.48}{0.96}{BurntOrange}{}{}
\FeatureStack{d1}{(3.48,0.50)}{0.58}{1.12}{BurntOrange}{}{}
\FeatureStack{d0}{(4.18,1.72)}{0.70}{1.28}{BurntOrange}{}{}
\node[head] (head) at (5.67,2.42) {$1{\times}1$};
\node[tinylabel, below=0.12cm of head] {sigmoid};

% Upright output image stack
\node[imagecard] (out1) at (7.00,1.92)
  {\includegraphics[width=1.30cm,height=1.62cm]{figures/report_images/prediction/prediction_0_908ccf4cb56e55558686.png}};
\node[imagecard] (out2) at (7.22,2.08)
  {\includegraphics[width=1.30cm,height=1.62cm]{figures/report_images/prediction/combined_0_f919a479b44de7fb61b4.png}};
\node[tinylabel] at (7.12,0.78) {output stack};
\node[tinylabel] at (7.12,0.50) {probability | TP/FP/FN};

% Main U-shaped flow
\draw[mainarrow] (img3.east) -- (e0west);
\draw[mainarrow] (e0south) -- (e1north);
\draw[mainarrow] (e1south) -- (e2north);
\draw[mainarrow] (e2south) -- (e3north);
\draw[mainarrow] (e3east) -- (bnwest);
\draw[mainarrow] (bneast) -- (d3west);
\draw[mainarrow] (d3north) -- (d2south);
\draw[mainarrow] (d2north) -- (d1south);
\draw[mainarrow] (d1north) -- (d0south);
\draw[mainarrow] (d0east) -- (head.west);
\draw[mainarrow] (head.east) -- (out1.west);

% Attention-gated skip connections
\node[gate] (a0) at (0.00,2.36) {attn0};
\node[gate] (a1) at (0.00,1.20) {attn1};
\node[gate] (a2) at (0.00,0.00) {attn2};
\node[gate] (a3) at (0.00,-1.20) {attn3};

\draw[skiparrow] (e0east) -- (a0.west);
\draw[gatearrow] (a0.east) -- (d0west);
\draw[skiparrow] (e1east) -- (a1.west);
\draw[gatearrow] (a1.east) -- (d1west);
\draw[skiparrow] (e2east) -- (a2.west);
\draw[gatearrow] (a2.east) -- (d2west);
\draw[skiparrow] (e3east) -- (a3.west);
\draw[gatearrow] (a3.east) -- (d3west);

\end{tikzpicture}%
}
\caption{U-shaped stacked implementation schematic for the Attention U-Net. Real report panels from \filename{16\_20250731\_Plate1\_A1\_D14\_MAX\_Crop.tif} are kept upright and stacked at the input and output. Editable TikZ feature stacks show the ResNet34 encoder, bottleneck, attention-gated skip connections, decoder, and sigmoid output head.}
\label{fig:architecture-detail}
\end{figure}
\end{landscape}
"""


def write_report_context(train_paths: dict, pred_paths: dict) -> None:
    context = f"""# Vessel Segmentation Report Context

This file is the compact implementation brief used to generate `main.tex`.

## Source runs
- Training run: W&B `eeebashit/vessel-seg/gsutr3pf`, Explorer job `6670898`.
- Training hardware: NVIDIA A100-SXM4-40GB, CUDA 12.1, PyTorch 2.5.1+cu121.
- Prediction run: local W&B run `run-20260509_162223-qi7txuwi`, Explorer job `6685455`.
- Prediction hardware: Tesla V100-SXM2-32GB, CUDA 12.1, PyTorch 2.5.1+cu121.

## Local report artifacts
- Training media: `figures/wandb/training/` ({sum(1 for _ in TRAIN_DIR.rglob('*.png'))} PNG files).
- Prediction media: `figures/wandb/prediction/` ({sum(1 for _ in PRED_DIR.rglob('*.png'))} PNG files).
- Downsampled report images: `figures/report_images/`.
- Logs and split file: `source_logs/`.
- Training manifest: `figures/manifest_training.csv`.
- Prediction manifest: `figures/manifest_prediction.csv`.

## Validation visualization mapping
"""
    for fold, info in FOLD_INFO.items():
        context += (
            f"- Fold {fold}: `{info['tiff']}`, split `validation`, "
            f"epochs {info['epochs']}, best epoch {info['best_epoch']}, "
            f"validation loss {info['val_loss']:.4f}.\n"
        )

    context += "\n## Final prediction mapping\n"
    for index, tiff, split, held_out in PRED_INFO:
        dice = f", Dice {TEST_DICE[tiff]:.4f}" if tiff in TEST_DICE else ""
        context += (
            f"- Index {index}: `{tiff}`, split `{split}`, "
            f"held out from training: {'yes' if held_out else 'no'}{dice}.\n"
        )

    context += """
## Metrics reported
- 5-fold cross-validation mean +/- std: 0.0651 +/- 0.0058.
- Selected fold: 1, best epoch 48, validation loss 0.0568.
- Test loss: 0.0662.
- Tile-level test Dice: 0.9158.
- Mean stitched-image Dice: 0.9189.
- Stitched Dice std: 0.0164.
- 95% CI: [0.8962, 0.9417].
- All 5 held-out test images reached Dice >= 0.90.
"""
    (ROOT / "REPORT_CONTEXT.md").write_text(context)


def build_main_tex(train_paths: dict, pred_paths: dict) -> str:
    lines = [
        r"\documentclass[11pt,letterpaper]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{charter}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage[dvipsnames]{xcolor}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{pdflscape}",
        r"\usepackage{caption}",
        r"\usepackage{subcaption}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta,positioning,shadows.blur,fit,shapes.geometric,calc}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\usepackage{xurl}",
        r"\usepackage[colorlinks=true,linkcolor=MidnightBlue,urlcolor=MidnightBlue,citecolor=MidnightBlue]{hyperref}",
        r"\usepackage{bookmark}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.65em}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\newcommand{\filename}[1]{\nolinkurl{#1}}",
        r"\title{\bfseries Publication-Ready Evidence Report\\Blood-Vessel Segmentation in 3D Fluorescence Microscopy}",
        r"\author{Abdullah Al Bashit}",
        r"\date{Training job 6670898 and prediction job 6685455}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        "This report documents a transfer-learning Attention U-Net workflow for segmenting blood-vessel structures in noisy 3D fluorescence microscopy images. "
        "The study focuses on weak contrast, uneven focus, fragmented vessels, and limited annotations. "
        "The final model used a trainable pretrained ResNet34 encoder, attention-gated decoder skips, engineered image channels, and a Tversky-style loss. "
        "Across five held-out test images, the stitched-image Dice score was 0.9189 on average, and every held-out test image achieved Dice at least 0.90.",
        r"\end{abstract}",
        r"\tableofcontents",
        r"\clearpage",
        r"\section{Introduction}",
        "Blood-vessel segmentation in fluorescence microscopy is difficult because the visual signal is not uniformly strong. "
        "Some vessels are bright and continuous, while others are faint, locally out of focus, or broken into short fragments. "
        "The goal of this work is therefore not only to train a segmentation model, but to understand how the model learns to recover weak vascular structure from noisy image evidence.",
        "The report is written as an evidence package. It includes the dataset representation, model design, loss function, training protocol, validation attention progressions, final test comparisons, and failure-mode visualizations. "
        "Every validation and final-result figure identifies the original TIFF filename and split type so that the evidence can be audited image by image.",
        r"\section{Dataset and Input Representation}",
        "The annotated dataset contained 25 image-mask pairs from day-7, day-14, and day-21 fluorescence microscopy acquisitions. "
        "The run used 20 images for training and cross-validation and 5 held-out images for the final test set. "
        "A separate prediction pass visualized 7 images: 2 train/validation pool images and all 5 held-out test images.",
        r"The model input at each pixel was a three-channel stack",
        r"\begin{equation}",
        r"\mathbf{x}(u,v) = \left[I(u,v),\, G(u,v),\, S(u,v)\right],",
        r"\end{equation}",
        "where $I$ is the normalized grayscale fluorescence intensity, $G$ is an edge-strength channel, and $S$ is a local sharpness channel. "
        "These two engineered channels give the network explicit access to vessel boundaries and focus-related texture without requiring a much larger model to rediscover those cues from intensity alone.",
        r"The gradient magnitude channel was computed from Sobel derivatives:",
        r"\begin{equation}",
        r"G(u,v) = \sqrt{\left(\partial_x I(u,v)\right)^2 + \left(\partial_y I(u,v)\right)^2}.",
        r"\end{equation}",
        r"The sharpness channel used local variation in the Laplacian response:",
        r"\begin{equation}",
        r"S(u,v) = \operatorname{Var}_{\Omega(u,v)}\left(\nabla^2 I\right),",
        r"\end{equation}",
        "where $\\Omega(u,v)$ is a local window around the pixel. High values indicate locally crisp structures; low values indicate blurred or texture-poor regions.",
        r"\section{Preprocessing and Tiling}",
        "The full images were processed as full-height horizontal strips. Each tile was 1300 pixels high and 1024 pixels wide, with a horizontal stride of 512 pixels. "
        "This design preserved long vertical context while keeping GPU memory use manageable. During stitched inference, overlapping tile predictions were blended with a Hann weighting map, optionally modulated by the sharpness channel so that locally reliable regions contributed more strongly to the final image-level prediction.",
        r"\section{Model Architecture}",
        "The model was an Attention U-Net with a trainable pretrained ResNet34 encoder. "
        "Decoder skip connections were gated before fusion, allowing the model to suppress irrelevant encoder features while preserving vessel-like structures. "
        "A lightweight dilated refinement head produced the final logit map, which was converted to vessel probability by a sigmoid function.",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\resizebox{0.98\linewidth}{!}{%",
        r"\begin{tikzpicture}[font=\small, box/.style={draw, rounded corners=2pt, minimum width=2.6cm, minimum height=1.05cm, align=center, fill=#1, blur shadow={shadow blur steps=5}}, arr/.style={-{Latex[length=3mm]}, thick}]",
        r"\node[box=BlueGreen!18] (input) {3-channel input\\$I, G, S$};",
        r"\node[box=SkyBlue!22, right=1.0cm of input] (enc1) {ResNet34\\encoder};",
        r"\node[box=SkyBlue!30, right=0.8cm of enc1] (enc2) {deep feature\\pyramid};",
        r"\node[box=Orange!22, right=1.0cm of enc2] (gate) {attention-gated\\skip features};",
        r"\node[box=YellowOrange!22, right=0.9cm of gate] (dec) {U-Net\\decoder};",
        r"\node[box=RedOrange!18, right=0.9cm of dec] (refine) {dilated\\refinement};",
        r"\node[box=ForestGreen!18, right=0.9cm of refine] (out) {vessel\\probability};",
        r"\draw[arr] (input) -- (enc1);",
        r"\draw[arr] (enc1) -- (enc2);",
        r"\draw[arr] (enc2) -- (gate);",
        r"\draw[arr] (gate) -- (dec);",
        r"\draw[arr] (dec) -- (refine);",
        r"\draw[arr] (refine) -- (out);",
        r"\draw[arr, dashed, MidnightBlue] (enc1.north) to[out=45,in=135] node[above, align=center] {skip features\\filtered by gates} (gate.north);",
        r"\draw[arr, dashed, MidnightBlue] (enc2.south) to[out=-45,in=-135] node[below, align=center] {semantic context} (dec.south);",
        r"\end{tikzpicture}%",
        r"}",
        r"\caption{Editable architecture schematic. The model receives intensity, gradient, and sharpness channels, processes them with a trainable pretrained ResNet34 encoder, filters skip features using attention gates, decodes vessel structure, refines the logits, and outputs a sigmoid vessel-probability map.}",
        r"\label{fig:architecture}",
        r"\end{figure}",
        detailed_architecture_figure(),
        "Figure~\\ref{fig:architecture-detail} labels the gated skip modules as Attn0--Attn3. "
        "Attn1--Attn3 correspond to the attention panels shown later; Attn0 is the finest-scale stem gate. "
        "Each gated skip is added to the decoder state before the next upsampling step.",
        r"The attention gate can be written as",
        r"\begin{equation}",
        r"\alpha_l = \sigma\left(\psi_l^\top \operatorname{ReLU}(W_x x_l + W_g g_l + b_l)\right), \qquad \tilde{x}_l = \alpha_l \odot x_l,",
        r"\end{equation}",
        "where $x_l$ is the encoder skip feature, $g_l$ is the decoder gating signal, and $\\tilde{x}_l$ is the filtered skip feature passed to the decoder.",
        r"\section{Loss Function and Optimization}",
        "The final run used a Tversky-style vessel loss. For predicted probability $p_i$ and binary mask $y_i$, the soft true-positive, false-positive, and false-negative terms are",
        r"\begin{equation}",
        r"TP=\sum_i p_i y_i,\qquad FP=\sum_i p_i(1-y_i),\qquad FN=\sum_i (1-p_i)y_i.",
        r"\end{equation}",
        r"The loss was",
        r"\begin{equation}",
        r"\mathcal{L}_{\mathrm{Tv}} = 1 - \frac{TP+\epsilon}{TP+\alpha FP+\beta FN+\epsilon},",
        r"\end{equation}",
        "with the configured final run using $\\beta=0.5$ and the Tversky term as the active segmentation loss. "
        "The model probability was $p_i=\\sigma(z_i)$, where $z_i$ is the output logit.",
        r"\section{Training Protocol}",
        "Training ran on the Explorer HPC cluster as job 6670898 using an NVIDIA A100-SXM4-40GB GPU, CUDA 12.1, and PyTorch 2.5.1+cu121. "
        "The optimizer was AdamW with learning rate $10^{-4}$ and weight decay $10^{-4}$, followed by cosine annealing to $10^{-6}$. "
        "The maximum training budget was 100 epochs, batch size was 12, and early stopping was applied independently within each fold with patience 30. "
        "Automatic mixed precision and gradient clipping were used during CUDA training. The final prediction pass ran as job 6685455 on a Tesla V100-SXM2-32GB GPU.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Training and inference configuration.}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Item & Value \\",
        r"\midrule",
        r"Training job & 6670898 on Explorer HPC \\",
        r"Training GPU & NVIDIA A100-SXM4-40GB \\",
        r"Prediction job & 6685455 on Explorer HPC \\",
        r"Prediction GPU & Tesla V100-SXM2-32GB \\",
        r"Software & CUDA 12.1, PyTorch 2.5.1+cu121 \\",
        r"Cross-validation & 5 folds \\",
        r"Batch size & 12 \\",
        r"Optimizer & AdamW, learning rate $10^{-4}$, weight decay $10^{-4}$ \\",
        r"Scheduler & Cosine annealing to $10^{-6}$ \\",
        r"Early stopping & patience 30, maximum 100 epochs \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:training-info}",
        r"\end{table}",
        r"\section{Results}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Five-fold validation summary. Fold 1 was selected for final held-out evaluation.}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Fold & Best epoch & Validation loss & Status \\",
        r"\midrule",
    ]
    for fold, info in FOLD_INFO.items():
        status = "selected" if info["selected"] else ""
        lines.append(f"{fold} & {info['best_epoch']} & {info['val_loss']:.4f} & {status} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:fold-results}",
            r"\end{table}",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Final held-out test-set performance.}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
            r"Cross-validation mean $\pm$ std & $0.0651 \pm 0.0058$ \\",
            r"Test loss & 0.0662 \\",
            r"Tile-level test Dice & 0.9158 \\",
            r"Mean stitched-image Dice & 0.9189 \\",
            r"Stitched-image Dice std & 0.0164 \\",
            r"95\% confidence interval & [0.8962, 0.9417] \\",
            r"Held-out test images with Dice $\ge 0.90$ & 5/5 \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:test-results}",
            r"\end{table}",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\begin{tikzpicture}",
            r"\begin{axis}[width=0.92\linewidth,height=7cm,xlabel={Epoch},ylabel={Validation loss},legend pos=north east,grid=both]",
        ]
    )
    colors = ["MidnightBlue", "RedOrange", "ForestGreen", "Purple", "TealBlue"]
    for fold, color in zip(FOLD_INFO, colors):
        lines.append(
            rf"\addplot[{color}, thick] table[x=epoch,y=val_loss]{{figures/plots/fold{fold}_curves.dat}};"
        )
        lines.append(rf"\addlegendentry{{Fold {fold}}}")
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            rf"\caption{{Editable PGFPlots validation-loss curves parsed from {tex_file('train_6670898.out')}. Fold 1 achieved the lowest validation loss and was used for final held-out inference.}}",
            r"\label{fig:val-curves}",
            r"\end{figure}",
            r"\section{Attention Progression During Training}",
            "The following validation figures show how the model's vessel probability and attention gates changed during training. "
            "Each figure explicitly names the validation TIFF, fold, and epochs shown. The attention overlays are not final test evidence; they are fold-level validation snapshots used to inspect learning behavior.",
        ]
    )
    for fold, info in FOLD_INFO.items():
        lines.append(rf"\subsection{{Fold {fold} Validation Image}}")
        lines.append(
            rf"{tex_file(info['tiff'])} is the fold {fold} validation image. "
            "The progression shows the model moving from broad vessel-like responses toward cleaner vessel masks as the validation loss stabilizes. "
            "The three attention-gate rows expose which decoder stages are emphasizing high-confidence vessel regions and which stages still respond to ambiguous background structure."
        )
        lines.append(training_progress_figure(train_paths, fold))
        lines.append(attention_only_figure(train_paths, fold))
    lines.extend(
        [
            r"\section{Final Error Analysis}",
            "The next figures are final prediction comparisons. Each stacked panel uses the same order and color convention: original image, gradient magnitude, sharpness map, ground truth, prediction, true positive, false positive, false negative, and combined TP/FP/FN overlay. "
            "Green denotes correctly recovered vessel pixels, red denotes false positives, and yellow denotes missed vessel pixels.",
        ]
    )
    for index, tiff, split, held_out in PRED_INFO:
        split_phrase = "held-out test image" if held_out else "train/validation pool image"
        lines.append(rf"\subsection{{Final Comparison {index + 1}}}")
        lines.append(
            rf"{tex_file(tiff)} is a {split_phrase}. "
            "The figure caption states the split type and whether the image was held out from training so that final test evidence is not mixed with train/validation pool visualization."
        )
        lines.append(final_prediction_figure(pred_paths, index, tiff, split, held_out))
    lines.extend(
        [
            r"\section{Discussion}",
            "The model performed well across the held-out test set, with all five test images exceeding Dice 0.90. "
            "The strongest qualitative behavior is recovery of long vessel structures despite weak contrast and local fragmentation. "
            "The main residual error modes are small false-positive structures in textured background regions and false negatives along extremely faint or broken vessel segments.",
            "The attention visualizations make the model behavior more interpretable than a single final mask. "
            "Early epochs often show diffuse attention, while later epochs concentrate attention on vessel-like structure. "
            "This supports the idea that the engineered channels and attention gates together help the model separate vascular signal from local blur, background texture, and annotation-limited ambiguity.",
            r"\section{Conclusion}",
            "This report packages the segmentation model, training protocol, quantitative results, and visual evidence into a publication-ready record. "
            "The final selected model achieved a mean stitched-image Dice of 0.9189 on the held-out test set, with consistent performance across all five test images. "
            "The evidence suggests that a compact, interpretable Attention U-Net can segment faint and fragmented vascular structures when intensity information is paired with gradient and sharpness cues.",
            r"\section{Future Directions}",
            r"\begin{itemize}",
            r"\item Expand annotation coverage across additional plates, days, imaging sessions, and biological replicates.",
            r"\item Add topology-aware losses such as clDice or skeleton-density penalties to improve continuity of faint vessel branches.",
            r"\item Quantify graph-level vascular connectivity after segmentation, including branch count, vessel length, and fragmentation metrics.",
            r"\item Perform external validation on microscopy sessions not used during model development.",
            r"\item Extend the pipeline toward full 3D volumetric segmentation if z-stack annotations become available.",
            r"\end{itemize}",
            r"\clearpage",
            r"\appendix",
            r"\section{Artifact Manifests}",
            "The CSV manifests generated with this report are saved as \\filename{figures/manifest_training.csv} and \\filename{figures/manifest_prediction.csv}. "
            "They list the W\\&B source file, report image file, original TIFF filename, split type, fold, epoch, panel type, and a short description for each figure panel used in the manuscript.",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    train_paths = training_lookup()
    pred_paths = prediction_lookup()
    write_training_curves()
    write_manifests(train_paths, pred_paths)
    write_report_context(train_paths, pred_paths)
    (ROOT / "main.tex").write_text(build_main_tex(train_paths, pred_paths))


if __name__ == "__main__":
    main()
