import utils.paths as paths
import utils.helpers as helpers
import os
from ultralytics import YOLO
from utils.arg_parser import parse_args  
from baseline_dataset import RockDetectionDataset, CombineRGBHillshade, ReplaceRGBChannelWithHS, NonLinearHSBlackout, MultiplyRGBHS, PCAFusion

"""
Train a supervised YOLOv11 model using RGB+hillshade fusion inputs.

The script:
1. Applies a transform (e.g. ReplaceRGBChannelWithHS)
2. Loads and splits the dataset into train/val/test (without overlap)
3. Prepares YOLO training files and config
4. Trains a model and evaluates it on validation and test sets

Example usage:
python src/train_baseline.py --model src/yolo11m_single_class.yaml --project runs/baseline_rgb_hs --bbox_size 32 --rgb_channel 1
"""

def main():
    # Parse command-line arguments
    args = parse_args()

    # TRANSFORM
    # transform = CombineRGBHillshade(alpha=args.alpha)
    transform = ReplaceRGBChannelWithHS(channel=args.rgb_channel)
    # transform = NonLinearHSBlackout(threshold=args.threshold)
    # transform = MultiplyRGBHS()
    # transform = PCAFusion()

    # LOAD DATA
    dataset = RockDetectionDataset(json_file=paths.JSON_FILE,
                                   root_dir=paths.RAW_DATA_DIR,
                                   transform=transform)
    
    # SPLIT DATA 
    train_samples, val_samples, test_samples = helpers.split_without_overlap(dataset, seed=42)

    # PREPARE TRAINING FILES
    helpers.prepare_yolo_training_files_all_splits(
        train_samples, 
        val_samples, 
        test_samples, 
        patch_size=args.imgsz,
        bbox_width=args.bbox_size,
        bbox_height=args.bbox_size)

    # LOAD MODEL
    model = YOLO(args.model)

    # CONFIG FILE CREATION
    root_processed_path = paths.PROCESSED_DATA_DIR
    train_relative_path = "images/train"
    val_relative_path = "images/val"
    test_relative_path = "images/test"  
    config_path = paths.BASELINE_CONFIG_YAML
    helpers.create_config_yaml(root_processed_path, train_relative_path, val_relative_path, test_relative_path, config_path)

    # TRAIN MODEL
    results = model.train(
        data=str(config_path),         
        epochs=args.epochs,
        time=args.time,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        pose=args.pose,
        kobj=args.kobj,
        plots=args.plots,
        translate=args.translate,
        scale=args.scale,
        flipud=args.flipud,
        fliplr=args.fliplr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        resume=args.resume,
        single_cls=args.single_cls,
        imgsz=args.imgsz,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        erasing=args.erasing,
        degrees=args.degrees,
        seed = args.seed,
        cache = False
    )

    # EXTRACT METRICS ON VALIDATION
    results_val = model.val(
        split="val", 
        project=args.project + "_val",
        )
    val_metrics_output_dir = os.path.join(args.project + "_val")
    helpers.save_metrics(results_val, val_metrics_output_dir, args.name)

    # EVALUATE MODEL ON TEST
    results_test = model.val(
        split="test", 
        project=args.project + "_test",
        )
    
    # EXTRACT TEST METRICS
    test_metrics_output_dir = os.path.join(args.project + "_test")
    helpers.save_metrics(results_test, test_metrics_output_dir, args.name)

if __name__ == '__main__':
    main()