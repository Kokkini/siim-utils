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
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import copy
import torch

from typing import Any, Dict, List

THING_CLASSES = ["negative", "typical", "indeterminate", "atypical", "negWithBox"]
CLASS_TO_ID = dict([(cls_name, id) for id, cls_name in enumerate(THING_CLASSES)])

# asign the study labels to all the bounding boxes
def get_study_image_mapping(indir):
    study_to_images = defaultdict(list)
    image_to_study = {}
    for study in tqdm(os.listdir(indir)):
        study_path = os.path.join(indir, study)
        for series in os.listdir(study_path):
            series_path = os.path.join(study_path, series)
            for img in os.listdir(series_path):
                if not img.endswith(".dcm"): continue
                img_id, ext = os.path.splitext(img)
                study_to_images[study].append(img_id)
                image_to_study[img_id] = study
    return study_to_images, image_to_study

def get_study_to_label_mapping(study_csv):
    df = pd.read_csv(study_csv)
    study_to_label = {}
    for i in tqdm(range(len(df))):
        id = df["id"][i]
        if id.endswith("image"): continue
        id = id.split("_")[0]
        item = {}
        item["negative"] = int(df["Negative for Pneumonia"][i])
        item["typical"]  = int(df["Typical Appearance"][i])
        item["indeterminate"] = int(df["Indeterminate Appearance"][i])
        item["atypical"] = int(df["Atypical Appearance"][i])
        study_to_label[id] = item
    return study_to_label
        
def create_new_detection_labels(image_csv, study_csv, new_image_csv, image_to_study):
    image_df = pd.read_csv(image_csv)
    study_to_label = get_study_to_label_mapping(study_csv)
    classes = ["negative", "typical", "indeterminate", "atypical"]
    for c in classes:
        image_df[c] = 0    
    for i in tqdm(range(len(image_df))):
        id = image_df["id"][i]
        if id.endswith("study"): continue
        id = id.split("_")[0]
        study_id = image_to_study[id]
        label = study_to_label[study_id]
        for c in classes:
            image_df.at[i, c] = label[c]
    image_df.to_csv(new_image_csv, index=False)
    return image_df

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
                        "bbox_mode": BoxMode.XYWH_ABS,
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
                        "bbox": [b['x']*portion, b['y']*portion, b['width']*portion, b['height']*portion],
                        "bbox_mode": BoxMode.XYWH_ABS,
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
        dataset_dict.append(item)
    return dataset_dict

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def resize(array, portion, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    H, W = array.shape[:2]
    width = int(W*portion)
    height = int(H*portion)
    im = im.resize((width, height), resample)    
    return im

def dicom_to_png(dicom_dir, save_dir, portion):
    os.makedirs(save_dir, exist_ok=True)
    for dirname, _, filenames in tqdm(os.walk(dicom_dir)):
        for file in filenames:
            # set keep_ratio=True to have original aspect ratio
            xray = read_xray(os.path.join(dirname, file))
            im = resize(xray, portion)
            save_name = os.path.splitext(file)[0] #remove extension
            save_name = f'{save_name}.png'
            im.save(os.path.join(save_dir, save_name))

def format_pred(labels, boxes, scores) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        if label==CLASS_TO_ID["negative"]:
            labelstr='none 1 0 0 1 1'
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