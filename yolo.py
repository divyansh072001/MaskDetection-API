import numpy as np
import cv2
import os
import random

random.seed(186)


yolo_dir = os.path.abspath(os.path.dirname(__file__))


def init_net(config_filepath, weights_filepath, classnames_filepath, calculation_mode):

    net = cv2.dnn.readNet(config_filepath, weights_filepath, 'darknet')

    if calculation_mode == "gpu-fp16":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    elif calculation_mode == "gpu-fp32":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    else:
        pass
        # print("Using CPU")

    out_names = net.getUnconnectedOutLayersNames()

    with open(classnames_filepath, 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    return net, out_names, class_names


def warm_up_net(net, out_names, yolo_width, yolo_height):

    batch_size = 1

    frame = (np.random.standard_normal([yolo_width, yolo_height, 3]) * 255).astype(np.uint8)
    images = []
    for i in range(batch_size):
        images.append(frame)

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImages(images, size=(yolo_width, yolo_height), swapRB=True, crop=False, ddepth=cv2.CV_8U)

    # Run a model
    net.setInput(blob, scalefactor=1/255.0)
    outs = net.forward(out_names)

    # return nothing


def classname_color_mapping(names):
    if len(names) <= 10:
        custom_colors = [(255, 105, 180), (124, 252, 0), (0, 191, 255), (255, 255, 0), (255, 0, 0), (139, 0, 139), (255, 140, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255)]
        classname_color = {name:  custom_colors[i] for i, name in enumerate(names)}
    else:
        classname_color = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}
    return classname_color


def print_detections(detections):
    print("\nObjects:")
    for classname, confidence, box in detections:
        x, y, w, h = box
        print(f'{classname}: {confidence}    (left_x: {x}   top_y:  {y}   width:   {w}   height:  {h})')


def draw_boxes(image, detections, classname_color=dict()):
    font_scale = 0.5
    thickness = 2

    for obj_det in detections:
        classname, confidence, box = obj_det

        color = classname_color.get(classname, (255, 0, 0))  # deafult color is blue

        x, y, w, h = box

        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{classname}: {confidence:.2f}"
        

        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height), baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)
        text_offset_x = x+2
        text_offset_y = y + h - 5 # y+h-7
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))

        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        cv2.putText(image, text, (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        # (x+w+3, y+h-10)
    return image


def yolo_postprocess(outs, confidence_threshold, nms_threshold, frame_height, frame_width, class_names):

    ## if lastLayer.type == 'Region': ########
    list_class_id, list_confidence, list_box = [], [], []

    out_bboxes = outs[:, 0:4]
    # out_bbox_confs = outs[:, 4]
    out_class_confs = outs[:, 5:]

    out_class_confs_max_ind = np.argmax(out_class_confs, axis=1)
    out_class_confs_max = out_class_confs[np.arange(len(out_class_confs)), out_class_confs_max_ind]

    # Get conf_max mask
    out_class_confs_max_mask = out_class_confs_max[:] >= confidence_threshold

    # Remove un-masking bboxes, conf_max (class score), list_class_id
    out_bboxes_cut = out_bboxes[out_class_confs_max_mask, :]
    out_class_confs_max_cut = out_class_confs_max[out_class_confs_max_mask]
    out_class_confs_max_ind_cut = out_class_confs_max_ind[out_class_confs_max_mask]

    for idx in range(out_bboxes_cut.shape[0]):
        center_x = int(out_bboxes_cut[idx, 0] * frame_width)
        center_y = int(out_bboxes_cut[idx, 1] * frame_height)
        width = int(out_bboxes_cut[idx, 2] * frame_width)
        height = int(out_bboxes_cut[idx, 3] * frame_height)
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)
        list_box.append([left, top, width, height])

    list_class_id += out_class_confs_max_ind_cut.tolist()
    list_confidence += out_class_confs_max_cut.tolist()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    indices = []
    list_class_id = np.array(list_class_id)
    list_box = np.array(list_box)
    list_confidence = np.array(list_confidence)
    unique_classes = set(list_class_id)
    for cl in unique_classes:
        class_indices = np.where(list_class_id == cl)[0]
        conf = list_confidence[class_indices]
        box = list_box[class_indices].tolist()
        nms_indices = cv2.dnn.NMSBoxes(box, conf, confidence_threshold, nms_threshold)
        nms_indices = nms_indices[:, 0] if len(nms_indices) else []
        indices.extend(class_indices[nms_indices])


    final_list_class_id = [list_class_id[i] for i in indices]
    final_list_confidence = [list_confidence[i] for i in indices]
    final_list_box = [list_box[i] for i in indices]

    final_list_classname = [class_names[cid] for cid in final_list_class_id]


    detections = list(zip(final_list_classname, final_list_confidence, final_list_box))

    return detections

#---------------------
 
def yolo_load(model_name="common", calculation_mode="cpu"):


    config_filepath = os.path.join(yolo_dir, f"yolo_models/{model_name}.cfg")
    weights_filepath = os.path.join(yolo_dir, f"yolo_models/{model_name}.weights")
    classnames_filepath = os.path.join(yolo_dir, f"yolo_models/{model_name}.names")
    # ---------------------------------------------------------------------------------

    if not os.path.exists(config_filepath):
        raise ValueError("Invalid config filepath: " + os.path.abspath(config_filepath))

    if not os.path.exists(weights_filepath):
        raise ValueError("Invalid weights filepath: " + os.path.abspath(weights_filepath))

    if not os.path.exists(classnames_filepath):
        raise ValueError("Invalid classnames filepath: " + os.path.abspath(classnames_filepath))

    if calculation_mode not in ['cpu', 'gpu-fp32', 'gpu-fp16']:
        raise ValueError("calculation_mode should be one from {'cpu', 'gpu-fp32', 'gpu-fp16'}")
    
    # ---------------------------------------------------------------------------------

    # Init network
    net, out_names, class_names = init_net(config_filepath, weights_filepath, classnames_filepath, calculation_mode)

    yolo_model_params = net, out_names, class_names

    return yolo_model_params
    
def yolo_detect(yolo_model_params, input_filepath, confidence_threshold=0.50, yolo_width=608, yolo_height=608, save_output=False, nms_threshold=0.6):

    net, out_names, class_names = yolo_model_params

    classname_color = classname_color_mapping(class_names)

    # ----------------------------------------------------------------------------
    if isinstance(input_filepath, list):

        all_detections = list()

        for fp in input_filepath:
            if not os.path.exists(fp):
                raise ValueError("Invalid image input filepath: " + os.path.abspath(fp))

            output_filepath = fp + "-yolo-output.jpg"

            frame = cv2.imread(fp)

            frame_height, frame_width = frame.shape[:2]

            # frame.shape (1200, 1500, 3) and blob.shape  (1, 3, 608, 608)

            # ------------------------------------------------

            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (yolo_width, yolo_height), 0, swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(out_names)
            # outs : [ 12288 x 85,  3072 x 85,  768 x 85 ]

            outs_ = np.concatenate(outs, axis=0)
            # outs_ : 16128 x 85
            detections = yolo_postprocess(outs_, confidence_threshold, nms_threshold, frame_height, frame_width, class_names)
            
            # -------------------------------------------------------
            ## PLAY AREA
            all_detections.append(detections)

            # -------------------------------------------------------

            if save_output:
                frame = draw_boxes(frame, detections, classname_color=classname_color)
                cv2.imwrite(output_filepath, frame)


            # ---------------------------------------------------

        return all_detections


    elif isinstance(input_filepath, str):  # video
        print("not implemented for video input....................")
        raise

    else:
        print("Invalid type of input_filepath")
        raise




    
def image_yolo_detect(yolo_model_params, input_filepath, confidence_threshold=0.50, yolo_width=608, yolo_height=608, save_output=False, nms_threshold=0.6):

    net, out_names, class_names = yolo_model_params

    classname_color = classname_color_mapping(class_names)

    # ----------------------------------------------------------------------------
    if isinstance(input_filepath, list):

        all_detections = list()

        for fnum, frame in enumerate(input_filepath):

            output_filepath = f"{fnum+1}-yolo-output.jpg"

            if frame is not None: # and frame_height and frame_width:

                frame_height, frame_width = frame.shape[:2]
               
                # frame.shape (1200, 1500, 3) and blob.shape  (1, 3, 608, 608)

                # ------------------------------------------------

                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (yolo_width, yolo_height), 0, swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(out_names)

                outs_ = np.concatenate(outs, axis=0)
                detections = yolo_postprocess(outs_, confidence_threshold, nms_threshold, frame_height, frame_width, class_names)
                # -------------------------------------------------------
           

                # -------------------------------------------------------

                if save_output:
                    frame = draw_boxes(frame, detections, classname_color=classname_color)
                    cv2.imwrite(output_filepath, frame)


            # ---------------------------------------------------

            else:
                # print("input image does not seems to be an image")
                detections = list()

            
            #-------------------------------------------

            ## PLAY AREA
            all_detections.append(detections)

            #------------------------------


        return all_detections

    elif isinstance(input_filepath, str):  # video
        print("not implemented for video input....................")
        raise

    else:
        print("Invalid type of input_filepath")
        raise

# ---------------------------------------------------------
