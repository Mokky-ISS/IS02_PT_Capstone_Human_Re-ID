import multiprocessing as mp
import os
from threading import Thread
from human_tracker import camera_capture
from database import ImageDB
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import pandas as pd
import signal, sys
import datetime

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/', 'path to input video or set to 0 for webcam')
#flags.DEFINE_string('output', './outputs/', 'path to output video')
flags.DEFINE_boolean('output', False, 'path to output video')
flags.DEFINE_string('output_format', 'MJPG', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('db', True, 'save information in database')
flags.DEFINE_boolean('trajectory', False, 'draw historical trajectories on every tracked human')
flags.DEFINE_integer('input_skip_frame', 8, 'number of frame to be skipped')
flags.DEFINE_integer('db_skip_frame', 8, 'number of frame to be skipped')
flags.DEFINE_boolean('saliant_sampling', True, 'select and store unique frame only into database')
flags.DEFINE_boolean('plot_graph', False, 'plot graph for soft threshold')
flags.DEFINE_integer('parallel_ps', 2, 'number of human tracker process to run')
flags.DEFINE_boolean('online', False, 'run online image extraction using rtsp')
flags.DEFINE_boolean('reid', False, 'set to True to run with REID, set to False if new labelled data are needed to be recorded')

def db_process():
    pass
    #db = ImageDB()
    # db.insert_data

class MultiPs():
    def __init__(self):
        self.job = []
        self.thread = []

        # shared resource
        self.db_queue = mp.Queue()
        self.manager = mp.Manager()
        self.unique_id = self.manager.list()

    def log_msg(self):
        mp.log_to_stderr()
        logger = mp.get_logger()
        logger.setLevel(logging.DEBUG)

    def new_job(self, name, target, *args):
        j = mp.Process(name=name, target=target, args=args)
        j.daemon = True
        self.job.append(j)

    def new_thread(self, name, target, *args):
        t = Thread(name=name, target=target, args=args)
        t.daemon = True
        self.thread.append(t)

    def signal_handler(self, sig, frame):
        print('Main Program: You pressed Ctrl+C!')
        for j in self.job:
            j.join()

        sys.exit(0)

def cam_stream(mps):
    mps.job.clear()
    mps.new_job('camera_ch' + FLAGS.video, camera_capture, int(FLAGS.video))
    for j in mps.job:
        j.start()
    for j in mps.job:
        j.join()


def sequential_run(batch, db_path, mps):
    mps.job.clear()
    print("batch:", batch)
    for ch in batch:
        mps.new_job('camera_ch' + ch, camera_capture, int(ch), db_path)
    for j in mps.job:
        j.start()
    for j in mps.job:
        j.join()

def online_run(rtsp, cam, db_path, mps):
    mps.job.clear()
    for i in range(FLAGS.parallel_ps):
        # cam[i]:int , rtsp[i]:str
        mps.new_job('camera_ch' + str(cam[i]), camera_capture, cam[i], rtsp[i], db_path)
        print("New online process for cam " + str(cam[i]))
    for j in mps.job:
        j.start()
    for j in mps.job:
        j.join()  

def get_rtsp(file):
    table = pd.read_excel(file, dtype={'Camera RTSP Stream': str,  'Channel': int}, engine='openpyxl')
    return table

def create_ps_list(vfile):
    ch_list = []
    for f in vfile:
        filename = os.path.splitext(f)[0]
        if filename.split('ch')[0] == '' and filename.split('ch')[-1].isdigit() == True:
            print(filename.split('ch')[-1])
            ch_list.append(filename.split('ch')[-1])
    if len(ch_list) == 0:
        print("No video file with 'ch' name. Please rename your input video with 'ch[channel number].mp4'.")
        return -1
    ch_list.sort(key=int)
    ps_list = None
    last_ps_num = len(ch_list) % FLAGS.parallel_ps
    if last_ps_num != 0:
        last_ps = ch_list[-last_ps_num:]
        print("last_ps:", last_ps)
        first_ps = ch_list[:-last_ps_num]
        print("first_ps:", first_ps)
        ps_list = np.asarray(first_ps).reshape(-1, FLAGS.parallel_ps).tolist()
        ps_list.append(last_ps)
        print(ps_list)
    else:
        ps_list = np.asarray(ch_list).reshape(-1, FLAGS.parallel_ps).tolist()
        print(ps_list)

    return ps_list


def main_single(_argv):
    # initialize database
    img_db = ImageDB()
    img_db.create_table()
    camera_capture(2)


def main(_argv):
    mps = MultiPs()
    signal.signal(signal.SIGINT, mps.signal_handler)

    # mps.log_msg()
    print("Parent Process PID: " + str(os.getpid()))
    print("Initialize database..")

    # initialize database
    db_path = "./database/Image_" + str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S")) + ".db"
    print("db_path main: ", db_path)
    img_db = ImageDB(db_path)
    img_db.delete_dbfile()
    img_db.create_table()

    # online mode
    if FLAGS.online:
        table = get_rtsp('data/rtsp/rtsp_cam.xlsx')
        online_run(table.to_dict('dict')['rtsp'], table.to_dict('dict')['cam'], db_path, mps)
    # offline mode
    else:
        if not FLAGS.video.isdigit():      
            # get video file info from video folder
            vfile = os.listdir(FLAGS.video)
            if len(vfile) == 0:
                print("No files in the " + FLAGS.video)
                return -1

            ps_list = create_ps_list(vfile)

            print("Start Multiprocessing..")
            # run new camera process
            for batch in ps_list:
                sequential_run(batch, db_path, mps)
        else:
            cam_stream(mps)
    #mps.new_job('database_ps', db_process)
    # for j in mps.job:
    #    j.start()
    # for j in mps.job:
    #    j.join()

    print("End of program.")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
