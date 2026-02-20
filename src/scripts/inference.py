import sys
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO

"""
Run inference with a YOLOv11 model and (optionally) store side-by-side
raw + annotated images.

Example usage:
python src/inference.py --model models/baseline_best.pt --source data/inference/processed --output runs/inference --conf 0.10 --iou 0.40
"""

def main() -> None:
    parser = argparse.ArgumentParser("YOLOv11 inference")
    parser.add_argument("--model", required=True,
                        help="Path to .pt weights")
    parser.add_argument("--source", required=True,
                        help="Image or folder of images")
    parser.add_argument("--output", default="results",
                        help="Folder to save YOLO outputs")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40,
                        help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--device", default=0,
                        help="'cpu', 'cuda:0', etc.")
    args = parser.parse_args()

    # ---------------------- sanity-checks -----------------------------
    model_path  = Path(args.model)
    source_path = Path(args.source)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    if not source_path.exists():
        sys.exit(f"Source not found: {source_path}")

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------- inference -----------------------------
    model = YOLO(model_path)
    if device != "cpu":
        model.to(device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running inference …")
    model.predict(
        source=source_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=False,
        project=out_dir,
        name="predict",
        verbose=True,
        show_conf=True,
    )
    print(f"Annotated images saved to {out_dir/'predict'}")

if __name__ == "__main__":
    main()
