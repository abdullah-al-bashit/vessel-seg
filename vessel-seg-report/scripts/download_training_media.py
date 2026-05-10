from pathlib import Path

import wandb


RUN_PATH = "eeebashit/vessel-seg/gsutr3pf"
OUT_DIR = Path("/home/a.bashit/vessel_seg/report_export/training_media")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run = wandb.Api().run(RUN_PATH)
    files = [item for item in run.files() if item.name.startswith("media/images/fold")]

    skipped = 0
    downloaded = 0
    for item in files:
        target = OUT_DIR / item.name
        if target.exists():
            skipped += 1
            continue
        item.download(root=str(OUT_DIR), replace=False)
        downloaded += 1

    png_count = sum(1 for _ in OUT_DIR.rglob("*.png"))
    print(
        f"total={len(files)} skipped={skipped} "
        f"downloaded_now={downloaded} png={png_count}"
    )


if __name__ == "__main__":
    main()
