import numpy as np
from math import ceil
from constants import strides, min_boxes

# Function to define anchor box priors based on image size
def define_img_size(image_size):
    # Initialise lists for feature map and shrinkage
    shrinkage_list = []
    feature_map_w_h_list = []
    
    # Calculate feature map sizes for different image sizes
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    # Set up shrinkage list based on feature map sizes
    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    
    # Generate anchor box priors
    priors = generate_priors(
        feature_map_w_h_list, shrinkage_list, image_size, min_boxes
    )
    return priors

# Function to generate anchor box priors
def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    # Initialise list for anchor box priors
    priors = []
    
    # Iterate over feature maps
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        
        # Iterate over positions in the feature map
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                # Iterate over different box sizes
                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)

# Function for Hard Non-Maximum Suppression
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    # Extract scores and boxes from input
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    
    # Iterate over candidate boxes
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        
        # Break if reached the top-k or last box
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        
        # Calculate IoU and filter boxes
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    
    # Return picked boxes after NMS
    return box_scores[picked, :]

# Function to calculate area of a bounding box
def area_of(left_top, right_bottom):
    # Calculate width and height of the box
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

# Function to calculate Intersection over Union (IoU) between two sets of boxes
def iou_of(boxes0, boxes1, eps=1e-5):
    # Calculate overlap between left-top and right-bottom corners
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:]) 
    
    # Calculate overlap area, area of both boxes, and IoU
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

# Function to convert predicted locations to bounding boxes
def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    
    # Convert locations to bounding boxes using priors
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)

# Function to convert bounding box coordinates from center-form to corner-form
def center_form_to_corner_form(locations):
    return np.concatenate(
        [locations[..., :2] - locations[..., 2:] / 2,
         locations[..., :2] + locations[..., 2:] / 2], 
        len(locations.shape) - 1
    )