## UNCOMMEND THESE FLAGS IF YOU WANT TO DIRECTLY RUN THIS CODE USING python human_tracker.py
# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_integer('skip_frame', 30, 'number of frame to be skipped')
# flags.DEFINE_integer('cam_id', 15, 'camera ID to run on different camera')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_string('video', './data/video/', 'path to input video or set to 0 for webcam')
# flags.DEFINE_string('output', './outputs/', 'path to output video')
# flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.50, 'score threshold')
# flags.DEFINE_boolean('dont_show', False, 'dont show video output')
# flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
# flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
# flags.DEFINE_boolean('db', True, 'save information in database')

###
# Note: Attempt to import packages here will cause CUDA ERROR, as parent process (main.py) will call this script 
# and import packages into parent process instead of child process.
###

# absl flag packages
from absl.flags import FLAGS
from absl import app, flags, logging

# child process from main.py
def camera_capture(cam_id):
    # assign camera id into new camera process.
    flags.DEFINE_integer('cam_id', cam_id, 'camera ID to run on different camera')
    try:
        app.run(run_human_tracker)
    except SystemExit:
        pass
        
def run_human_tracker(_argv):
    # Every neccessary packages must be imported within the child process instead of parent process, 
    # to avoid error of "could not retrieve CUDA device count: CUDA_ERROR_NOT_INITIALIZED: initialization error"

    # export to database
    from database import ImageDB
    # deep sort imports
    from tools import generate_detections as gdet
    from deep_sort.tracker import Tracker
    from deep_sort.detection import Detection
    from deep_sort import preprocessing, nn_matching
    # system packages
    import time
    import os
    # tensorflow packages
    import tensorflow as tf
    from tensorflow.compat.v1 import InteractiveSession
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.python.saved_model import tag_constants
    import core.utils as utils
    from core.config import cfg
    from core.yolov4 import filter_boxes
    # other packages
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from PIL import Image

    # comment out below line to enable tensorflow logging outputs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
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
    vid_name = 'ch' + str(FLAGS.cam_id) + '.mp4'
    video_path = os.path.join(FLAGS.video, vid_name)
    print('video_path:', video_path)

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

    if FLAGS.db:
        img_db = ImageDB()
        img_db.create_table()

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
        output_name = 'ch' + str(FLAGS.cam_id) + '-tracked.mp4'
        output_path = os.path.join(FLAGS.output, output_name)
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        print('output_path:', output_path)

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        print("[cam %d] Frame #: %d" %(FLAGS.cam_id,frame_num))

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
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
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
        #allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

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
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # copy a new frame for patch image without boxes
        patch_frame = frame.copy()

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # update database
            # skip frame is done here to extract less data for database, if overall FPS for videocapture is reduced to one, tracker wont work. 
            if FLAGS.db and frame_num % FLAGS.skip_frame == 0:
                print("======== DATABASE =========")
                # single patch box
                patch_bbox = track.to_tlwh()
                patch_np = gdet.get_img_patch(patch_frame, patch_bbox)
                patch_np = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
                # https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/
                is_success, im_buf_arr = cv2.imencode(".jpg", patch_np)
                patch_img = im_buf_arr.tobytes()

                # export data to database
                img_db.insert_data(FLAGS.cam_id, track.track_id, patch_img, patch_np, patch_bbox, frame_num)
                #print("Data Type:", type(frame_num),type(track.track_id),type(patch_img),type(patch_bbox))

        # draw bbox on screen
            color_num = ''.join(str(ord(c)) for c in track.track_id)
            color = colors[int(color_num) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("[cam %d] FPS: %.2f" % (FLAGS.cam_id, fps))
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_num += 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
