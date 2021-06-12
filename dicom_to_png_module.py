from PIL import Image
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os
from tqdm.auto import tqdm

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
