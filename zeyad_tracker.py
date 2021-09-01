# Run yolov4 deep sort object tracker on video
# python save_model.py --weights ./data/yolov4-obj_best.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4

#################LIGHT##################
# python zeyad_tracker.py --video ./data/video/LightSegment1.mp4 --output ./outputs/Light/Test/LightTestOutput.avi --model yolov4
# python zeyad_tracker.py --video ./data/video/LightSegment2.mp4 --output ./outputs/Light/Segment2/LightSegment2Output.avi --model yolov4
# python zeyad_tracker.py --video ./data/video/LightSegment3.mp4 --output ./outputs/Light/Segment3/LightSegment3Output.avi --model yolov4

#################SNOW##################
# python zeyad_tracker.py --video ./data/video/SnowSegment1.mp4 --output ./outputs/Snow/Segment1/SnowSegment1Output.avi --model yolov4
# python zeyad_tracker.py --video ./data/video/SnowSegment2.mp4 --output ./outputs/Snow/Segment2/SnowSegment2Output.avi --model yolov4
# python zeyad_tracker.py --video ./data/video/SnowSegment3.mp4 --output ./outputs/Snow/Segment3/SnowSegment3Output.avi --model yolov4
# python zeyad_tracker.py --video ./data/video/SnowSegment4.3gp --output ./outputs/Snow/Segment4/SnowSegment4Output.avi --model yolov4

import os
import sys

import time
import tensorflow as tf
import pickle

import each_side_tracking_v02

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils

pts = []

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', 'C:/Users/zeyad/Desktop/InputDesktop/LightSegment1.mp4',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', 'C:/Users/zeyad/Desktop/OutputDesktop/LightSegment1_OUTPUT.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.3, 'iou threshold')
flags.DEFINE_float('score', 0.1, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_list('coordinates', 'str', 'coordinates')


def main(_argv):
    # Definition of the parameters
    print(sys.argv)
    print(FLAGS.coordinates)

    # change it to 0.5
    max_cosine_distance = 0.4
    nn_budget = None

    # change it to 0.8
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    # DEFINING A DICTIONARY FOR TRACKING
    track_dict = dict()
    frame_list = []

    from _collections import deque

    # change 2048 to higher value
    pts = [deque(maxlen=10240) for _ in range(1000000)]

    counter_up = []

    ped_up_list = []

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.1,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()  # based on Kalman Filtering
        tracker.update(detections)

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # update tracks
        centers = []
        id_centers = []
        for track in tracker.tracks:
            # if Kalman filtering does not assign a track, keep going
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            #################  TRACKING OPERATION ####################

            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            id_center = [str(track.track_id), (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))]

            # update the points queue
            centers.append(center)
            # print(id_centers)
            id_centers.append(id_center)
            # print(id_centers)
            pts[track.track_id].append(center)

            x0 = int(FLAGS.coordinates[0])
            x1 = int(FLAGS.coordinates[1])
            x2 = int(FLAGS.coordinates[2])
            x3 = int(FLAGS.coordinates[3])
            y0 = int(FLAGS.coordinates[4])
            y1 = int(FLAGS.coordinates[5])
            y2 = int(FLAGS.coordinates[6])
            y3 = int(FLAGS.coordinates[7])

            line1 = cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
            line2 = cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 0), thickness=2)
            line3 = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
            line4 = cv2.line(frame, (x0, y0), (x3, y3), (0, 0, 0), thickness=1)

            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                X = pts[track.track_id][j - 1][0]
                Y = pts[track.track_id][j - 1][1]
                # Limit the amount of computation, by only taking orthogonality of lines within the min and max values
                if (min(x0, x1, x2, x3) <= X <= max(x0, x1, x2, x3)) and (
                        min(y0, y1, y2, y3) <= Y <= max(y0, y1, y2, y3)):
                    c = (int(X) - x1) * (y2 - y1)
                    d = (int(Y) - y1) * (x2 - x1)
                    side1 = c - d
                    m = (int(X) - x0) * (y3 - y0)
                    n = (int(Y) - y0) * (x3 - x0)
                    side2 = m - n
                    # if the id's position is between the tracking region,
                    # then there will be one positive and one negative orthogonal statement
                    if (side1 >= 0 and side2 <= 0) or (side2 >= 0 and side1 <= 0):
                        a = (int(X) - x0) * (y1 - y0)
                        b = (int(Y) - y0) * (x1 - x0)
                        enter1 = a - b
                        q = (int(X) - x3) * (y2 - y3)
                        p = (int(Y) - y3) * (x2 - x3)
                        enter2 = q - p

                        if (enter1 >= 0 and enter2 <= 0) or (enter2 >= 0 and enter1 <= 0):
                            # draw bbox on screen only for crosswalks
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                            cv2.circle(frame, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)), 2, color,
                                       -1)
                            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 3)), 0, 0.25,
                                        color, 1)

                            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                            cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color,
                                     1)  # thickness
                            counter_up.append(int(track.track_id))

                            ped_up = [int(track.track_id), pts[track.track_id][j]]
                            ped_up_list.append(ped_up)
                            # print(ped_up_list)
                else:
                    continue

            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                        int(bbox[0]),
                                                                                                        int(bbox[1]),
                                                                                                        int(bbox[2]),
                                                                                                        int(bbox[3]))))

            each_id_list = [frame_num, str(track.track_id), int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
            frame_list.append(each_id_list)

        # calculate frames per second of running detections
        total_up_count = len(set(counter_up))

        cv2.putText(frame, "Total Pedestrians Passing Sidewalk = " + str(total_up_count), (10, 50), 0, 0.5, (0, 255, 0),
                    1)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        # split output string at file type to get pickle file name
        a = FLAGS.output.split('.avi')
        output = a[0]
        with open(output + ".pickle", 'wb') as handle:
            pickle.dump(frame_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()
    each_side_tracking_v02.track_data(FLAGS.output, FLAGS.coordinates, ped_up_list)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
