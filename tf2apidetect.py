
"""
Object Detection From TF2 Saved Model
=====================================
"""
import os, sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import numpy as np
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model_dir = r"C:\Users\TME-DJ\tf2api\data\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"

label_path = r"C:\Users\TME-DJ\tf2api\data\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\mscoco_label_map.pbtxt"

path_saved_model = model_dir + "/saved_model"

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(path_saved_model)
category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

import warnings

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

# ----------------read image and test--------------------#
image_path = r"C:\Users\TME-DJ\tf2api\data\pic\1.png"
image_np = cv2.imread(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=0.30,
    agnostic_mode=False)

cv2.imshow("object_detection_demo", image_np_with_detections)
cv2.waitKey()
cv2.destroyAllWindows()
print('Done')
