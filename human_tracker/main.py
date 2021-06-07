import multiprocessing as mp
import os
from threading import Thread
from human_tracker import camera_capture
from database import ImageDB
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './outputs/', 'path to output video')
flags.DEFINE_string('output_format', 'MJPG', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('db', True, 'save information in database')
flags.DEFINE_boolean('trajectory', False, 'draw historical trajectories on every tracked human')
flags.DEFINE_integer('skip_frame', 30, 'number of frame to be skipped')
flags.DEFINE_boolean('saliant_sampling', True, 'select and store unique frame only into database')
flags.DEFINE_boolean('plot_graph', False, 'plot graph for soft threshold')
flags.DEFINE_integer('parallel_ps', 2, 'number of human tracker process to run')


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


def sequential_run(batch, mps):
    mps.job.clear()
    print("batch:", batch)
    for ch in batch:
        mps.new_job('camera_ch' + ch, camera_capture, int(ch))
    for j in mps.job:
        j.start()
    for j in mps.job:
        j.join()


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

    # mps.log_msg()
    print("Parent Process PID: " + str(os.getpid()))
    print("Initialize database..")
    # initialize database
    img_db = ImageDB()
    img_db.delete_dbfile()
    img_db.create_table()

    # get video file info from video folder
    vfile = os.listdir(FLAGS.video)
    if len(vfile) == 0:
        print("No files in the " + FLAGS.video)
        return -1

    ps_list = create_ps_list(vfile)

    print("Start Multiprocessing..")
    # run new camera process
    for batch in ps_list:
        sequential_run(batch, mps)

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
