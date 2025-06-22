# agents/vision_agent.py

import os
import json
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from typing import Dict


class VisionAgent:
    def __init__(
        self,
        label_map_path: str,
        confidence_threshold: float = 0.5,
        device: str = None,
    ):
        """
        Loads a COCO-pretrained Faster R-CNN model and a label map for annotation.

        Args:
            label_map_path: Path to a JSON file mapping COCO class indices → names.
            confidence_threshold: Minimum score to keep a detection.
            device: Optional override for "cpu" or "cuda". If None, auto-detects.
        """
        # --- Load label map ---
        if not os.path.isfile(label_map_path):
            raise FileNotFoundError(f"Label map not found: {label_map_path}")
        with open(label_map_path, "r") as f:
            # JSON keys are strings; convert them to int
            self.label_map: Dict[int, str] = {
                int(k): v for k, v in json.load(f).items()
            }

        # --- Determine device ---
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"VisionAgent: Using device = {self.device}")

        # --- Load COCO-pretrained Faster R-CNN (ResNet-50 FPN) ---
        # pretrained=True loads COCO weights (num_classes=91)
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        ).to(self.device)
        self.detector.eval()
        print("VisionAgent: Loaded COCO-pretrained Faster R-CNN (ResNet-50 FPN).")

        self.conf_thresh = confidence_threshold

        # --- Prepare font for annotations ---
        try:
            self.font = ImageFont.truetype("arial.ttf", size=16)
        except IOError:
            self.font = ImageFont.load_default()

    def run(self, image: Image.Image) -> Dict:
        """
        Detect objects and return both raw data and an annotated PIL image:
          - `objects`: [ { label, score, bbox }, ... ]
          - `annotated_image`: PIL.Image with colored boxes/text

        Expects:
          - `image`: a PIL.Image in RGB mode (or convertible via ToTensor).
        """
        # 1) Inference
        tensor = torchvision.transforms.ToTensor()(image).to(self.device)
        outputs = self.detector([tensor])[0]

        boxes = outputs["boxes"].detach().cpu().tolist()
        scores = outputs["scores"].detach().cpu().tolist()
        labels = outputs["labels"].detach().cpu().tolist()

        # 2) Filter by confidence
        kept = []
        for box, score, lbl in zip(boxes, scores, labels):
            if score < self.conf_thresh:
                continue
            label_name = self.label_map.get(lbl, str(lbl))
            kept.append({"label": label_name, "score": score, "bbox": box})

        if not kept:
            return {
                "error": "no_objects",
                "message": f"No objects detected above confidence {self.conf_thresh:.2f}.",
            }

        # 3) Annotate Image
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for obj in kept:
            x0, y0, x1, y1 = obj["bbox"]
            lbl_txt = obj["label"]
            scr_txt = f"{obj['score']:.2f}"

            # Red bounding box
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

            # Red label background & yellow text
            tw, th = draw.textsize(lbl_txt, font=self.font)
            draw.rectangle([x0, y0 - th, x0 + tw, y0], fill="red")
            draw.text((x0, y0 - th), lbl_txt, font=self.font, fill="yellow")

            # Yellow score below box
            draw.text((x0, y1 + 2), scr_txt, font=self.font, fill="yellow")

        return {"objects": kept, "annotated_image": annotated}
