from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict
import os

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