from yolo import yolo_load, yolo_detect
from glob import glob
import numpy as np

def detect_mask(img_path):
    img_list = glob(img_path)

    mask_model = yolo_load(model_name="mask")

    mask_detections = yolo_detect(mask_model, input_filepath=img_list, confidence_threshold=0.5, nms_threshold=0.3, yolo_width=512, yolo_height=512, save_output=False)

    outputs = {}

    if len(mask_detections) > 0:
        outputs['detections'] = {}
        outputs['detections']['labels'] = []
        for i in mask_detections:
            for j in i:
                detection = {}
                detection['label'] = j[0]
                detection['confidence'] = j[1]
                ls = j[2].tolist()
                detection['X'] = ls[0]
                detection['Y'] = ls[1]
                detection['W'] = ls[2]
                detection['H'] = ls[3]

                outputs['detections']['labels'].append(detection)
        
    else:
        outputs['detections'] = 'No Face Detected'

    return outputs