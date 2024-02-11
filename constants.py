import numpy as np

# Constants for pre-processing and detection
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [
    [10.0, 16.0, 24.0], 
    [32.0, 48.0], 
    [64.0, 96.0], 
    [128.0, 192.0, 256.0]
]
strides = [8.0, 16.0, 32.0, 64.0]
threshold = 0.5