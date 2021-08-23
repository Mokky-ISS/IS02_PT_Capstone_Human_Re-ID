# UNCOMMEND THESE FLAGS IF YOU WANT TO DIRECTLY RUN THIS CODE USING python human_tracker.py
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

db_queue = None

# child process from main.py
def camera_capture(*args):
    # assign camera id into new camera process.
    global db_queue
    # online
    if args[0]:
        print("Online mode")
        flags.DEFINE_integer('cam_id', args[1], 'camera ID to run on different camera')
        flags.DEFINE_string('rtsp', args[2], 'rtsp to run on different camera')
        flags.DEFINE_integer('gpu', args[3], 'gpu to run on different camera')
        flags.DEFINE_string('db_path', args[4], 'database save path.')
        print("db_path check: ", args[4])
        db_queue = args[5]
    # offline
    else:
        print("Offline mode")
        flags.DEFINE_integer('cam_id', args[1], 'camera ID to run on different camera')
        flags.DEFINE_integer('gpu', args[2], 'gpu to run on different camera')
        flags.DEFINE_string('db_path', args[3], 'database save path.')
        print("db_path check: ", args[3])
        db_queue = args[4]
    try:
        app.run(run_human_tracker)
        # In the app.run() argument, include tuple arguments after the target function to fill in the _argv  
        # e.g. app.run(run_human_tracker, ("test",))
    except SystemExit:
        pass

def run_human_tracker_2(_argv):
    print('rtsp: ' + FLAGS.rtsp)
    print('cam_id: ' + str(FLAGS.cam_id))

def run_human_tracker_1(_argv):
    import time
    import os
    import cv2

    vid_name = 'ch' + str(FLAGS.cam_id) + '.mp4'
    video_path = os.path.join(FLAGS.video, vid_name)
    print('video_path:', video_path)
    vid = cv2.VideoCapture(video_path)

    while (True):
        ret, frame = vid.read()
        cv2.imshow("camCapture", frame)
        # time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def run_human_tracker(_argv):
    # Every neccessary packages must be imported within the child process instead of parent process,
    # to avoid error of "could not retrieve CUDA device count: CUDA_ERROR_NOT_INITIALIZED: initialization error"
    
    # export to database
    from database import ImageDB
    # check human pose
    from pose_estimation import check_pose
    # deep sort imports
    from tools import generate_detections as gdet
    from deep_sort.tracker import Tracker
    from deep_sort.detection import Detection
    from deep_sort import preprocessing, nn_matching
    # reid imports
    from reid_inference import Reid
    # system packages
    import time
    import datetime as dt
    import os
    import multiprocessing as mp
    import sys, signal
    import shutil
    import schedule
    # tensorflow packages
    import tensorflow as tf
    from tensorflow.compat.v1 import InteractiveSession
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.python.saved_model import tag_constants
    import core.utils as utils
    from core.config import cfg
    # other packages
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from PIL import Image
    from collections import deque

    # comment out below line to enable tensorflow logging outputs
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print("physical device:", physical_devices)
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            #tf.config.experimental.set_visible_devices(gpus[0:1], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            print("FlagGPU: ", FLAGS.gpu)
            # online
            if FLAGS.online:
                tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[FLAGS.gpu], True)
            # offline 
            else:
                # if only single gpu is running, use gpus[0]. If two gpu are running, use FLAGS.gpu 
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
            #tf.config.set_logical_device_configuration(
            #    gpus[0],
            #    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("[Cam "+str(FLAGS.cam_id)+"]:", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print(gpus)
            print(logical_gpus)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])

    # Increase max_cosine_distance variable to reduce identity switching.
    # Increase the threshold will increase the tolerance to change the track id on a human.
    # Refer non_max_suppression function in preprocessing.py.
    # max_cosine_distance = 0.4 (original value)
    max_cosine_distance = 0.55
    # 0.55

    # Number of features to store in every tracked human.
    # Refer partial_fit function in nn_matching.py.
    #nn_budget = None
    nn_budget = 1000

    # Reduce this variable to reduce identity switching.
    # This will only reduce overlap detections within one frame,
    # to ensure that only one detection box is assigned to one human.
    # Refer non_max_suppression function in preprocessing.py.
    # nms_max_overlap = 1.0 (original value)
    nms_max_overlap = 0.75
    # 0.75

    # Saliant sampling soft threshold
    soft_thred = 0.08
    #soft_thred = 0.01

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    devices = session.list_devices()
    print("Session devices: ")
    for d in devices:
        print(d.name)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = None

    if FLAGS.online:
        video_path = FLAGS.rtsp
        print('video_path:', video_path)
        print('cam_id: ' + str(FLAGS.cam_id))
    else:
        if not FLAGS.video.isdigit():
            vid_name = 'ch' + str(FLAGS.cam_id) + '.mp4'
            video_path = os.path.join(FLAGS.video, vid_name)
            print('video_path:', video_path)
        else:
            video_path = int(FLAGS.video)
            print("Use webcam", video_path)
            print(type(video_path))

    # load model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    if FLAGS.db:
        img_db = ImageDB(FLAGS.db_path)
        print("db_name: ", FLAGS.db_path)
        #img_db = ImageDB("./database/Image_" + str(FLAGS.cam_id) + ".db")
        # img_db.delete_dbfile()
        # img_db.create_table()

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

    # Dictionary for historical trajectory points
    h_pts = {}
    #h_pts = [deque(maxlen=30) for _ in range(1000)]

    # plot soft threshold graph
    if FLAGS.plot_graph:
        x_list = {}
        y_list = {}
        fig = plt.figure()
        # 1x1 grid, first subplot
        ax = fig.add_subplot(1, 1, 1)

    # initialize reid (individual camera database)
    if FLAGS.reid:
        cam_path = FLAGS.cam_db_path + "/Cam_" + str(FLAGS.cam_id) + ".db"
        reid_db = ImageDB(db_name=cam_path)
        print("cam_path: ", cam_path)
        reid_db.delete_dbfile()
        reid_db.create_table() 
        reid = Reid(cam_path)

    def signal_handler(sig, frame):
        name = mp.current_process().name
        print(str(name) + ': You pressed Ctrl+C!')
        # not required to send signal back to database_ps thread, use signal_handler in database_ps.
        vid.release()
        if FLAGS.output:
            out.release()

        cv2.destroyAllWindows()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)

    save = True 
    reset = False
    frame_num = 0
    # while video is running
    while True:
        now = dt.datetime.now()
        t = now.timetuple()
        # t[6] consists of day name information. 0 = Monday. 4 = Friday.
        if t[6] > 4:
            if save:
                # copy the database with timestamp, and copy the db once only.
                #db_name = now.strftime("Reid_%Y%m%d.db")
                #db_filepath = os.path.join("../reid/database", db_name)
                #shutil.copy2(FLAGS.reid_db_path, db_filepath)
                #print("Reid database file is saved at: ", db_filepath)
                global db_queue
                db_queue.put(cam_path)
                save = False
                reset = True

            # during weekend, this tracker will sleep, and check for time every hour.
            print("Standby during weekend [", now.strftime("%A, %d. %B %Y %I:%M%p"), ']')   
            time.sleep(1*60*60)
            continue 
        else:
            if reset:
                # reset database record and reset it once
                reid_db.delete_dbfile()
                reid_db.create_table() 
                reset = False
                save = True       

        start_time = time.time()
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(frame)
        else:
            print('Video has ended.')
            break
        print("[cam %d] Frame #: %d" % (FLAGS.cam_id, frame_num))

        # Skip input frames
        if frame_num % FLAGS.input_skip_frame != 0:
            frame_num += 1
            continue

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # run detections
        # with mirrored_strategy.scope():
        # try:
        # with tf.device('/device:GPU:0'):
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
        # except RuntimeError as e:
        #     print(e)

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

        # filter blur boxes
        bbox_width_threshold = 60
        bbox_height_threshold = 120

        #print("bboxes:", bboxes)
        #print("original_h:", original_h)
        #print("original_w:", original_w)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(
            bboxes, scores, names, features) if bbox[2] > bbox_width_threshold and bbox[3] > bbox_height_threshold]

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

        if FLAGS.plot_graph:
            ax.clear()

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
            if FLAGS.db and frame_num % FLAGS.db_skip_frame == 0:
                # Saliant sampling
                print("=================================================")
                print("Active targets: ", end=" ")
                dist = None
                for sample_id, features in tracker.metric.samples.items():
                    print(sample_id, len(features), ";", end=" ")
                    if sample_id != track.track_id:
                        continue
                    last_feat = []
                    last_feat.append(features[-1])
                    if sample_id in tracker.metric.store_feats:
                        # cosine distance checking with soft threshold
                        # dist is 1x1 ndarray
                        #print("last_feat:", last_feat[0].shape)
                        #print("store_feats:", tracker.metric.store_feats[sample_id][0].shape)
                        dist = nn_matching._cosine_distance(np.asarray(last_feat), np.asarray(tracker.metric.store_feats[sample_id]))
                        # if dist is more than soft threshold, the track would have the same human with different view
                        if dist[0, 0] > soft_thred:
                            track.unique_same_human = True
                        # store the lastest feature into store_feats
                        tracker.metric.store_feats[sample_id] = [features[-1]]
                    else:
                        tracker.metric.store_feats.setdefault(sample_id, []).append(features[-1])
                        #print("unique_same_human:", sample_id)
                        track.unique_same_human = True
                print('\n')
                print("dist:", dist)
                if dist is not None and FLAGS.plot_graph:
                    trk_id = track.track_id
                    print("current track_id:", trk_id)
                    print("dist:", dist[0, 0])
                    x_list.setdefault(trk_id, []).append(frame_num)
                    y_list.setdefault(trk_id, []).append(dist[0, 0])
                    print("x_list:", x_list[trk_id])
                    print("y_list:", y_list[trk_id])
                    print("x len:", len(x_list[trk_id]))
                    print("y len:", len(y_list[trk_id]))
                    #ax.plot(x_list[trk_id], y_list[trk_id], label=trk_id)
                    # ax.set_yscale('log')

                # Record unique frame only into database
                if track.unique_same_human == True or not FLAGS.saliant_sampling:
                    print("======== DATABASE =========")
                    print("record track_id:", track.track_id)
                    # single patch box (patch_bbox is obtained from DeepSort without 0.5 aspect ratio)
                    patch_bbox = track.to_tlwh()
                    patch_np = gdet.get_img_patch(patch_frame, patch_bbox)

                    # Check for blurry image
                    patch_np_gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
                    score = cv2.Laplacian(patch_np_gray, cv2.CV_64F).var()
                    b_blur = True
                    if score <= 2500:
                        b_blur = False
                     
                    # Check for human image using Pose estimation
                    patch_np = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
                    b_pose = check_pose(patch_np, track.track_id)
                    
                    # https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/
                    is_success, im_buf_arr = cv2.imencode(".jpg", patch_np)
                    patch_img = im_buf_arr.tobytes()

                    if b_pose and b_blur:
                        if FLAGS.reid:
                            # run reid inference process
                            img_id = img_db.get_imgid(FLAGS.cam_id, track.track_id)
                            reid.run(img_id, patch_img)
                            #img_db.insert_data(FLAGS.cam_id, track.track_id, patch_img, patch_np)
                        else:
                            # export data to database
                            img_db.insert_data(FLAGS.cam_id, track.track_id, patch_img, patch_np)
                            #img_db.insert_data_old(FLAGS.cam_id, track.track_id, patch_img, patch_np, patch_bbox, frame_num, original_w, original_h)
                            #print("Data Type:", type(frame_num),type(track.track_id),type(patch_img),type(patch_bbox))

                    # Reset unique_same_human bool state
                    track.unique_same_human = False

            if not FLAGS.dont_show:
                # draw bbox on screen
                color_num = ''.join(str(ord(c)) for c in track.track_id)
                color = colors[int(color_num) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)

            # draw historical trajectory
            if FLAGS.trajectory:
                center = (int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2))
                h_pts.setdefault(track.track_id, deque(maxlen=30)).append(center)
                for j in range(1, len(h_pts[track.track_id])):
                    if h_pts[track.track_id][j-1] is None or h_pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64/float(j+1))*2)
                    cv2.line(frame, (h_pts[track.track_id][j-1]), (h_pts[track.track_id][j]), color, thickness)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        period = time.time() - start_time
        print("[cam %d] FPS: %.2f" % (FLAGS.cam_id, fps))
        print("[cam %d] Period: %.2f" % (FLAGS.cam_id, period))
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            window_name = "Output from cam " + str(FLAGS.cam_id)
            cv2.imshow(window_name, result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_num += 1

        # plot graph for soft threshold
        if FLAGS.plot_graph:
            if bool(x_list) and bool(y_list):
                key_list = x_list.keys() & y_list.keys()
                for k in key_list:
                    ax.plot(x_list[k], y_list[k], label=k)
                    ax.set_yscale('log')
                plt.xlabel('Frames')
                plt.ylabel('Cosine Distance (log)')
                plt.title('Soft threshold plot')
                # line for soft threshold
                plt.axhline(y=soft_thred, color='r', linestyle='-', label="Soft Threshold")
                plt.legend(loc="upper left")
                plt.pause(0.000001)
        print("========================= END =========================\n")
    if FLAGS.plot_graph:
        plt.savefig('soft_threshold.png')
    # plt.show()
    vid.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
