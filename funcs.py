from collections import defaultdict
from enum import auto
import os
from unicodedata import category
import pandas as pd
from tqdm.auto import tqdm
import detectron2

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode

import os
from tqdm.auto import tqdm
import pandas as pd
import ast
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import copy
import torch
import math

from typing import Any, Dict, List

THING_CLASSES = ["negative", "typical", "indeterminate", "atypical", "negWithBox"]
CLASS_TO_ID = dict([(cls_name, id) for id, cls_name in enumerate(THING_CLASSES)])


def parse_anno_csv(anno_csv):
    print('parsing anno csv')
    res = {}
    df = pd.read_csv(anno_csv)
    for i in tqdm(range(len(df))):
        id = df["id"][i]
        if id.endswith("study"): continue
        id = id.split("_")[0]
        boxes = df["boxes"][i]
        bboxes = []
        if pd.isna(boxes) or boxes.strip() == "":
            for c in THING_CLASSES[:-1]:
                # cases: negative, non negative
                # if it's non negative without boxes, don't do anything
                # if it's negative without boxes, add a big box in the middle labeled negative
                if c not in df.columns or df[c][i] != 1: continue
                category_id = CLASS_TO_ID[c]
                if c == "negative":
                    bboxes.append({
                        "bbox": [0,0,1,1],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": category_id
                    })
        else:
            boxes = ast.literal_eval(boxes.strip())
            for c in THING_CLASSES[:-1]:
                # cases: negative, non negative
                # if it's negative with boxes, then label those boxes as negWithBox instead of negative
                if c not in df.columns or df[c][i] != 1: continue
                category_id = CLASS_TO_ID[c]
                if c == "negative":
                    category_id = CLASS_TO_ID["negWithBox"]
                for b in boxes:
                    portion = 1/3
                    bboxes.append({
                        "bbox": [b['x']*portion, b['y']*portion, (b['x']+b['width'])*portion, (b['y']+b['height'])*portion],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": category_id
                    })
        res[id] = bboxes
    return res
        
def dataset_function(indir, anno_csv, auto_label=False, study_to_images=None, image_to_study=None):
    # auto_label: set True if you want to automatically draw boxes on unlabeled
    # images in the same study
    dataset_dict = []
    id_to_bboxes = parse_anno_csv(anno_csv)
    if auto_label:
        for id, bboxes in id_to_bboxes.items():
            if bboxes != []: continue
            study = image_to_study[id]
            study_images = study_to_images[study]
            for img_id in study_images:
                if id_to_bboxes[img_id] == []: continue
                id_to_bboxes[id] = copy.deepcopy(id_to_bboxes[img_id])
                break
    print("making anno")
    for file in tqdm(os.listdir(indir)):
        item = {}
        if not file.endswith(".png"): continue
        filepath = os.path.join(indir, file)
        id, ext = os.path.splitext(file)
        item["image_id"] = id
        item["file_name"] = filepath
        im = cv2.imread(filepath)
        H, W = im.shape[:2]
        item["width"] = W
        item["height"] = H
        item["annotations"] = id_to_bboxes[id]
        # the category can only be negative if there are no boxes
        # change the box for all negative instances
        for box in item["annotations"]:
            if box["category_id"] == CLASS_TO_ID["negative"]:
                box["bbox"] = [W*0.1, H*0.1, W*0.8, H*0.8]
        # make sure bboxes is within H, W
        for box in item["annotations"]:
            x1, y1, x2, y2 = box["bbox"]
            box["bbox"] = [max(x1, 0), max(y1, 0), min(x2, W-1), min(y2, H-1)]
        dataset_dict.append(item)
    return dataset_dict

def format_pred(labels, boxes, scores) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        if label==CLASS_TO_ID["negative"]:
            pred_strings.append('none 1 0 0 1 1') 
        else:
            labelstr='opacity'
            pred_strings.append(f"{labelstr} {score:0.3f} {xmin} {ymin} {xmax} {ymax}") 
    return " ".join(pred_strings)

def predict_batch(predictor, im_list):
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        for original_image in im_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # Do not apply original augmentation, which is resize.
            # image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)
        return predictions

def format_detectron_output(output, portion):
    instances = output["instances"]
    if len(instances) == 0:
        # No finding, let's set 2 1.0 0 0 1 1x. Negative
        result = {
            "negative": 1,
            "typical": 0,
            "indeterminate": 0,
            "atypical": 0,
            "PredictionString": "none 1 0 0 1 1"}
    else:
        # Find some bbox...
        # print(f"index={index}, find {len(instances)} bbox.")
        fields: Dict[str, Any] = instances.get_fields()
        pred_classes = fields["pred_classes"]  # (n_boxes,)
        pred_scores = fields["scores"]
        # shape (n_boxes, 4). (xmin, ymin, xmax, ymax)
        pred_boxes = fields["pred_boxes"].tensor

        pred_boxes[:, [0, 2]] /= portion
        pred_boxes[:, [1, 3]] /= portion

        pred_classes_array = pred_classes.cpu().numpy()
        pred_boxes_array = pred_boxes.cpu().numpy()
        pred_scores_array = pred_scores.cpu().numpy()
        pred_classes_scores_array=np.stack((pred_classes_array,pred_scores_array), axis=-1)
        
        result = {}
        for cls_name, cls_id in CLASS_TO_ID.items():
            result[cls_name] = np.mean(pred_classes_scores_array[pred_classes_scores_array[:,0]==cls_id, 1],axis=0)
        
        result["PredictionString"] = format_pred(pred_classes_array, pred_boxes_array, pred_scores_array)
    return result

def full_dataset_detection(predictor, dataset_dict, batch_size, portion=1/3):
    results_list = []
    for i in tqdm(range(math.ceil(len(dataset_dict) / batch_size))):
        inds = list(range(batch_size * i, min(batch_size * (i + 1), len(dataset_dict))))
        dataset_dicts_batch = [dataset_dict[i] for i in inds]
        im_list = [cv2.imread(d["file_name"]) for d in dataset_dicts_batch]
        outputs_list = predict_batch(predictor, im_list)
        for im, outputs, d in zip(im_list, outputs_list, dataset_dicts_batch):
            result = format_detectron_output(outputs, portion)
            result["image_id"] = d["image_id"]
            results_list.append(result)
    return results_list
