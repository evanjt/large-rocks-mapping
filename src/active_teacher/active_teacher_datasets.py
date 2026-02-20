import os
import cv2
import torch
from typing import List
from torch.utils.data import Dataset
from torchvision import transforms
from ultralytics.utils.ops import xyxy2xywhn
from utils.constants import TILE_SIZE_PX

"""
Custom datasets for semi-supervised Active Teacher training.

- UnlabeledDataset loads raw .tif images from a directory.
- PseudoLabelDataset stores top-N pseudo-labeled predictions for use in training.

These are used to dynamically manage the growing pseudo-labeled dataset across iterations.
"""

class UnlabeledDataset(Dataset):
        
    def __init__(self, img_dir: str, imgsz=TILE_SIZE_PX):
        # collect all img file paths
        self.img_paths: List[str] = [
            os.path.join(img_dir, fn)
            for fn in sorted(os.listdir(img_dir)) #sorted to ensure consistent order
            if fn.lower().endswith('.tif')
        ]
        self.imgsz = imgsz
        self.transform = transforms.ToTensor()
        print(f"Found {len(self.img_paths)} unlabelled images in '{img_dir}'.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # lazy load of images
        path = self.img_paths[idx]
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)
        return {'img': img_tensor, 'img_path': path}
    
    def pop_indices(self, indices: List[int]):
        for i in sorted(indices, reverse=True):
            self.img_paths.pop(i)
        print(f"Removed {len(indices)} images from the dataset. Remaining: {len(self.img_paths)}")

    def retrieve_topN_predictions(
        self,
        predictions: List[torch.Tensor],
        N: int
    ) -> tuple[
        List[str], List[torch.Tensor],  # top-N
        List[str], List[torch.Tensor]   # the rest
    ]:
        """
        Returns:
          top_img_paths:   List of paths for the top N entries.
          top_yolo_preds:  Their corresponding YOLO-formatted preds.
          other_img_paths: List of paths for the remaining entries.
          other_yolo_preds:Their corresponding YOLO-formatted preds.
        """

        # 1. Compute average confidence per image
        avg_confs = [
            (preds[:, 4].mean().item() if (preds is not None and preds.numel() > 0) else 0.0)
            for preds in predictions
        ]

        # 2. Sort all indices by descending avg conf
        sorted_inds = sorted(range(len(avg_confs)), key=lambda i: avg_confs[i], reverse=True)

        # 3. Split into top and other indices
        top_inds   = sorted_inds[:N]
        other_inds = sorted_inds[N:]

        # 4. Helper to convert one preds→yolo_preds
        def _to_yolo(preds: torch.Tensor):
            if preds is None or preds.numel() == 0:
                return torch.empty((0,5))
            # xyxy → normalized xywh
            boxes_xyxy = preds[:, :4]
            yolo_boxes = xyxy2xywhn(boxes_xyxy, w=self.imgsz, h=self.imgsz)
            cls_ids     = preds[:, 5].unsqueeze(1).long().to(yolo_boxes.dtype)
            return torch.cat((cls_ids, yolo_boxes), dim=1)

        # 5. Gather top-N
        top_img_paths  = [ self.img_paths[i] for i in top_inds ]
        top_yolo_preds = [ _to_yolo(predictions[i]) for i in top_inds ]

        # 7. Remove only the top-N from the dataset
        self.pop_indices(top_inds)

        return top_img_paths, top_yolo_preds
    

class PseudoLabelDataset(Dataset):

    def __init__(self, img_paths: list[str], preds: list[torch.Tensor], imgsz=TILE_SIZE_PX):
        assert len(img_paths) == len(preds), "images / preds length mismatch"
        # sort for deterministic indexing
        self.img_paths = img_paths
        self.preds = preds
        self.imgsz = imgsz
        print(f"Created {len(self.img_paths)} pseudo-labeled images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        bgr  = cv2.imread(path)
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img  = torch.from_numpy(rgb).permute(2, 0, 1) # Not ToTensor yet because not standardized in YOLODataset!! (no need to convert to float yet)
        pred = self.preds[idx] # [N,5]
        if pred.numel() == 0:
            # Make cls 2-D (0×1) to match YOLODataset
            cls_ids = torch.zeros((0, 1), dtype=torch.long)
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
        else:
            # Keep the extra dimension: [N,1], not [N]
            cls_ids = pred[:, 0:1].long()
            boxes   = pred[:, 1:].float()
        # batch_idx placeholders—remains 1-D
        return {
            'im_file':       path,
            'ori_shape':     (self.imgsz, self.imgsz),
            'resized_shape': (self.imgsz, self.imgsz),
            'img':           img,
            'cls':           cls_ids,
            'bboxes':        boxes,
            'batch_idx':     torch.zeros((cls_ids.size(0),), dtype=torch.long)
        }