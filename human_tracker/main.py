import multiprocessing as mp
import os
from threading import Thread
from human_tracker import camera_capture
from database import ImageDB
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './outputs/', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
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
    img_db.create_table()

    print("Start Multiprocessing..")
    # run new camera process
    #mps.new_job('camera_ch2', camera_capture, 2)
    #mps.new_job('camera_ch3', camera_capture, 3)
    #mps.new_job('database_ps', db_process)
    mps.new_job('camera_ch15', camera_capture, 15)

    for j in mps.job:
        j.start()

    for j in mps.job:
        j.join()

    print("End of program.")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
