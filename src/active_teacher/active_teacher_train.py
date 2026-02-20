import yaml
import torch
import pandas as pd
import argparse

from types import SimpleNamespace
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG as cfg
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.dataset import YOLOConcatDataset
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.torch_utils import ModelEMA

from active_teacher_datasets import UnlabeledDataset, PseudoLabelDataset
from utils.constants import TILE_SIZE_PX
from utils.helpers import validate_model

"""
Run semi-supervised training with the Active Teacher framework.

This script initializes a student-teacher loop where the teacher model generates
pseudo-labels for unlabelled data, which are iteratively added to the training set.

Example usage:
python src/active_teacher.py --batch-size 8 --workers 4 --iterations 20 --alpha 4.0 --N 100 --decay 0.9996 --lr 1e-5 --outmodel runs/new_al_experiments/active_teacher_last.pt --outcsv runs/new_al_experiments/metrics.csv
"""


def parse_args():
    p = argparse.ArgumentParser(description="Active Teacher Training")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--iterations", type=int, default=20, help="number of AL iterations")
    p.add_argument("--alpha", type=float, default=4.0, help="pseudo-label loss weight")
    p.add_argument("--N", type=int,   default=100, help="number of pseudo-labels per iteration")
    p.add_argument("--decay", type=float, default=0.9996)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--outmodel", type=str, default="runs/new_al_experiments/active_teacher_last.pt")
    p.add_argument("--outcsv", type=str, default="runs/new_al_experiments/metrics.csv")
    return p.parse_args()

# python src/active_teacher.py --batch-size 8 --workers 4 --iterations 20 --alpha 4.0 --N 100 --decay 0.9996 --lr 1e-5 --outmodel runs/new_al_experiments/active_teacher_last_alpa_4_test.pt --outcsv runs/new_al_experiments/metrics_alpa_4_test.csv

def main():

    # -- ARGS --
    args = parse_args()

    # -- CONFIG --
    cfg_path = 'src/active_teacher_cfg.yaml'
    # load dataset YAML into a dict
    with open(cfg_path) as f:
        data_dict = yaml.safe_load(f)

    # constants
    batch_size = args.batch_size
    workers = args.workers

    # override config
    cfg.batch = batch_size
    cfg.scale = 0 # the source of all my problems...
    cfg.flipud = 0.5

    # -- DATASET --
    # (train_ds[0] is a dict : ['im_file', 'ori_shape', 'resized_shape', 'img', 'cls', 'bboxes', 'batch_idx'])
    train_ds = build_yolo_dataset(cfg, 'data/processed/images/train', batch=cfg.batch, data=data_dict,
    mode='train')
    val_ds = build_yolo_dataset(cfg, 'data/processed/images/val', batch=cfg.batch, data=data_dict, mode='val')
    pseudo_ds = PseudoLabelDataset([], [])

    # unlabelled
    unlabeled_data_path = 'data/BO/processed'
    #unlabeled_data_path = 'data/processed/images/val'
    unlabeled_ds = UnlabeledDataset(unlabeled_data_path)

    # -- DATALOADER --
    train_dl = build_dataloader(train_ds, batch=batch_size, workers=workers, shuffle=True)
    val_dl = build_dataloader(dataset=val_ds, batch=cfg.batch, workers=workers, shuffle=False)

    # unlabelled dataloader
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,)

    # -- DEVICE --
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- MODEL --
    student = YOLO('src/yolo11m_single_class.yaml', task='detect', verbose=False).load('yolo11_rgb/exp_rgb_channel_1/weights/best.pt')

    student.model.nc = 1
    student.model.npx = 1

    student.model.names = {0 : 'rock'}
    student.model.args['batch'] = batch_size
    student.model.args['data'] = cfg_path
    student.model.args['optimizer'] = 'AdamW'
    student.model.args['imgsz'] = TILE_SIZE_PX
    student.model.args['single_cls'] = True
    student.model.args['box'] = 7.5
    student.model.args['cls'] = 0.5
    student.model.args['dfl'] = 1.5
    student.model.args = SimpleNamespace(**student.model.args) # CONVERT TO NAMESPACE OTHERWISE args NOT DEFINED
    print("Model arguments: ", student.model.args)

    # load model to device
    student.to(device)

    # -- OPTIMIZER --
    optimizer = AdamW(student.model.parameters(), lr=args.lr, weight_decay=0.0005) # TODO check lower lrs

    # --- EMA SETUP ---
    decay = args.decay
    teacher = ModelEMA(student.model, decay=decay) 

    # ----------------------------------------------------

    # -- TRAINING LOOP
    iterations = args.iterations
    alpha = args.alpha # weight for pseudo-labeled loss
    N = args.N # number of pseudo-labeled images to select at each iteration

    metrics_list = [] # store all metrics at each iteration

    for it in range(iterations):
        
        # ----------- PREDICTION ON UNLABELLED ------------
        teacher.ema.eval() #teacher model for inference
        all_preds = []
        
        with torch.no_grad():
            for batch in unlabeled_loader:
                # normalize & send to device
                imgs = batch['img'].float().to(device)
                raw_preds = teacher.ema(imgs)
                # apply NMS per image
                nmsed = non_max_suppression(raw_preds, conf_thres=0.15, iou_thres=0.40)
                all_preds.extend(nmsed)  # flatten into a single list
                #show_first_with_preds(imgs[0], nmsed[0])

        #print("Number of predictions per image in unlabelled:", [pred.shape[0] if pred is not None else 0 for pred in all_preds])

        # select top N images + their YOLO‐formatted preds, and remove them from unlabeled_ds
        top_img_paths, top_preds = unlabeled_ds.retrieve_topN_predictions(all_preds, N=N)

        # move tensor predictions to CPU to use less GPU mem
        top_preds  = [p.cpu() for p in top_preds]

        top_pseudo_ds = PseudoLabelDataset(top_img_paths, top_preds)
        print("length of unlabelled dataloader", len(unlabeled_loader))
        
        # ----------- UPDATE DATASETS ------------
        # add the top N pseudo-labeled images to the training dataset and rebuild dataloader
        pseudo_ds = YOLOConcatDataset([pseudo_ds, top_pseudo_ds])
        pseudo_dl = build_dataloader(dataset=pseudo_ds, batch=cfg.batch, workers=workers, shuffle=True)
        print("length of new pseudo dataloader", len(pseudo_dl))
        
        # ----------- UPDATE WEIGHTS ------------
        student.model.train()

        train_running_loss = 0.0
        train_runningloss_items = torch.zeros(3).to(device)

        for batch in train_dl:
            
            # convert all tensor fields to the right device & dtype
            batch['img'] = batch['img'].float().div(255.0).to(device)
            batch['batch_idx'] = batch['batch_idx'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)

            # Forward pass (loss items is cls, box, dfl losses)
            total_loss, loss_items = student.model(batch)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            teacher.update(student.model) # update teacher

            train_running_loss += total_loss.item()
            train_runningloss_items += loss_items.detach()


        pseudo_running_loss = 0.0
        pseudo_runningloss_items = torch.zeros(3).to(device)

        for batch in pseudo_dl:

            # convert all tensor fields to the right device & dtype
            batch['img'] = batch['img'].float().div(255.0).to(device)
            batch['batch_idx'] = batch['batch_idx'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)

            # Forward pass (loss items is cls, box, dfl losses)
            total_loss, loss_items = student.model(batch)
            weighted_loss = total_loss * alpha

            # Backward and optimize
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            teacher.update(student.model) # update teacher

            pseudo_running_loss += total_loss.item()
            pseudo_runningloss_items += loss_items.detach()
            

        # ---------- VALIDATION ---------- 
        student.model.eval()

        val_running_loss = 0.0
        val_running_items = torch.zeros(3, device=device)

        with torch.no_grad():
            for batch in val_dl:
                # exactly the same pre‐processing as in training:
                batch['img'] = batch['img'].float().div(255.0).to(device)
                batch['batch_idx'] = batch['batch_idx'].to(device)
                batch['cls'] = batch['cls'].to(device)
                batch['bboxes'] = batch['bboxes'].to(device)

                # forward only — returns total_loss and the per‐term breakdown
                loss, items = student.model(batch)

                val_running_loss  += loss.item()
                val_running_items += items

        # loss statistics
        avg_loss = train_running_loss / (len(train_dl) * batch_size)
        avg_loss_items = train_runningloss_items / len(train_dl)

        # pseudo loss
        avg_pseudo_loss = pseudo_running_loss / (len(pseudo_dl) * batch_size)
        avg_pseudo_items = pseudo_runningloss_items / len(pseudo_dl)
        
        # val loss
        avg_val_loss  = val_running_loss / (len(val_dl) * batch_size)
        avg_val_items = val_running_items / len(val_dl)

        print(
        f"Epoch {it+1}/{iterations} — "
        f"Train Loss:  {avg_loss:.4f}  "
        f"(box={avg_loss_items[0]:.4f}, cls={avg_loss_items[1]:.4f}, dfl={avg_loss_items[2]:.4f})  |  "
        f"Pseudo Loss: {avg_pseudo_loss:.4f}  "
        f"(box={avg_pseudo_items[0]:.4f}, cls={avg_pseudo_items[1]:.4f}, dfl={avg_pseudo_items[2]:.4f})  |  "
        f"Val Loss:    {avg_val_loss:.4f}  "
        f"(box={avg_val_items[0]:.4f}, cls={avg_val_items[1]:.4f}, dfl={avg_val_items[2]:.4f})"
        )

        val_metrics_teacher = validate_model(teacher.ema, val_dl, device=device, conf_thres=0.1, iou_thres=0.4) #p, r, f1, f2
        val_metrics_student = validate_model(student.model, val_dl, device=device, conf_thres=0.1, iou_thres=0.4) #p, r, f1, f2

        metrics = {
            "iter": it + 1,
            "train/loss": avg_loss,
            "train/box": avg_loss_items[0].item(),
            "train/cls": avg_loss_items[1].item(),
            "train/dfl": avg_loss_items[2].item(),
            "pseudo/loss": avg_pseudo_loss,
            "pseudo/box": avg_pseudo_items[0].item(),
            "pseudo/cls": avg_pseudo_items[1].item(),
            "pseudo/dfl": avg_pseudo_items[2].item(),
            "val/loss": avg_val_loss,
            "val/box": avg_val_items[0].item(),
            "val/cls": avg_val_items[1].item(),
            "val/dfl": avg_val_items[2].item(),
        }
        # merge teacher metrics under a "teacher/" namespace
        metrics.update({ f"teacher/{k}": v for k, v in val_metrics_teacher.items() })
        # merge student metrics under a "student/" namespace
        metrics.update({ f"student/{k}": v for k, v in val_metrics_student.items() })
        metrics_list.append(metrics)

    # save model
    save_path = args.outmodel
    student.model.load_state_dict(teacher.ema.state_dict())
    student.save(save_path)
    print(f"EMA model saved to {save_path}")

    # save metrics
    save_metrics_path = args.outcsv
    df = pd.DataFrame(metrics_list).round(4)
    df.to_csv(save_metrics_path, index=False)
    print(f"Metrics saved to {save_metrics_path}")

if __name__ == "__main__":
    main()