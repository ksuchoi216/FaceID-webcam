import torch
import cv2
from PIL import Image
# DeepSORT -> Importing DeepSORT.
from .ref_code.deep_sort.application_util import preprocessing
from .ref_code.deep_sort.deep_sort import nn_matching
from .ref_code.deep_sort.deep_sort.detection import Detection
from .ref_code.deep_sort.deep_sort.tracker import Tracker
from .ref_code.deep_sort.tools import generate_detections as gdet


from .ref_code.yolov5.models.common import DetectMultiBackend
from .ref_code.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from .ref_code.yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                         increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from .ref_code.yolov5.utils.plots import Annotator, colors, save_one_box
from .ref_code.yolov5.utils.torch_utils import select_device, time_sync


class ObjectTracker():
  def __init__(self, args):
    self.args = args
    self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance, args.nn_budget)
    tracker = Tracker(metric)

  def detect_objects(self, frame):
    img = Image.fromarray(frame)
    res = self.object_detector(img)
    # res.print()
    df = res.pandas().xyxy[0]
    df = df.loc[(df['name'] == 'person') & (df['confidence'] >= 0.7)]
    df = df[['xmin', 'ymin', 'xmax', 'ymax']]

    boxes = df.to_numpy()

    for i, box in enumerate(boxes):
      frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
    
    return frame

  def track_objects(self, frame):
    # DeepSORT -> Initializing tracker.
    max_cosine_distance = 0.4
    nn_budget = None
    model_filename = './mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
