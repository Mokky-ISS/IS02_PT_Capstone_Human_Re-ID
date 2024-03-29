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
import datetime as dt
import shutil
import time

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('rtsp_path', 'data/rtsp/rtsp_cam.xlsx', 'default rtsp camera path')
flags.DEFINE_string('reid_db_path', "../reid/archive", 'default reid database path, where all of the samples from cam are saved into the db with timestamp')
#flags.DEFINE_string('cam_db_path', '../reid/database', 'default cam database path')
#flags.DEFINE_string('merge_db_path', "../reid/database/merge", 'default merged reid database path, where all of the samples from cam are saved into the db with timestamp')
#flags.DEFINE_string('output', './outputs/', 'path to output video')
flags.DEFINE_boolean('output', False, 'path to output video')
flags.DEFINE_string('output_format', 'MJPG', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being trascked on screen')
#flags.DEFINE_boolean('db', True, 'save information in database')
flags.DEFINE_boolean('trajectory', False, 'draw historical trajectories on every tracked human')
flags.DEFINE_integer('input_skip_frame', 8, 'number of frame to be skipped')
flags.DEFINE_integer('db_skip_frame', 8, 'number of frame to be skipped')
flags.DEFINE_boolean('saliant_sampling', True, 'select and store unique frame only into database')
flags.DEFINE_boolean('plot_graph', False, 'plot graph for soft threshold')
flags.DEFINE_integer('parallel_ps', 8, 'number of human tracker process to run')
flags.DEFINE_boolean('online', True, 'run online image extraction using rtsp')
flags.DEFINE_boolean('reid', True, 'set to True to run with REID, set to False if new labelled data are needed to be recorded')

# def db_process(*args):
#     def signal_handler(sig, frame):
#         name = mp.current_process().name
#         print(str(name) + ': You pressed Ctrl+C!')

#         db_list = [] 
#         # args[0] is the camera list
#         # args[1] is the length of camera list
#         # args[2] is the cam path
#         # args[3] is the db merge path
#         # args[4] is the shared queue recording the database paths 
        
#         for i in range(args[1]):
#             if args[0]:
#                 db_list.append(args[2] + "/Cam_" + str(args[0][i]) + ".db")
#         # finish gathering the db_paths, run merge.
#         print('Saving merge database..')
#         now = dt.datetime.now()
#         db_name = now.strftime("Reid_Interrputed_%Y%m%d.db")
#         db_filepath = os.path.join(args[3], db_name)
#         reid_db = ImageDB(db_name=db_filepath)
#         reid_db.delete_dbfile()
#         reid_db.create_table()
#         reid_db.merge_data(db_list)

#         sys.exit(0)
        
#     signal.signal(signal.SIGINT, signal_handler)

#     while True:
#         db_list = [] 
#         # args[0] is the length of camera list
#         # args[1] is the shared queue recording the database paths 
#         while len(db_list) < args[1]:
#             print("db_path_process: ", args[4].get())
#             db_list.append(args[4].get())
#             #time.sleep(1)
#         # finish gathering the db_paths, run merge.
#         print('Saving merge database..')
#         now = dt.datetime.now()
#         db_name = now.strftime("Reid_%Y%m%d.db")
#         db_filepath = os.path.join(args[3], db_name)
#         reid_db = ImageDB(db_name=db_filepath)
#         reid_db.delete_dbfile()
#         reid_db.create_table()
#         reid_db.merge_data(db_list)
    


class MultiPs():
    def __init__(self):
        self.job = []
        self.thread = []
        self.cam = []

        # shared resource
        #self.db_queue = mp.Queue()
        self.manager = mp.Manager()
        self.stop_tracker = self.manager.Value('i',0)
        self.stop_main_ps = self.manager.Value('i',1)
        self.db_path = None
        self.unique_id = self.manager.list()

    def log_msg(self):
        mp.log_to_stderr()
        logger = mp.get_logger()
        logger.setLevel(logging.DEBUG)

    def new_job(self, name, target, *args):
        print("args: ", args)
        args = (*args, self.stop_main_ps, self.stop_tracker)
        print("new args: ", args)
        j = mp.Process(name=name, target=target, args=args)
        j.daemon = True
        self.job.append(j)

    def new_thread(self, name, target, *args):
        t = Thread(name=name, target=target, args=args)
        t.daemon = True
        self.thread.append(t)

    def create_new_db(self):
        #global db_path
        #global reid
        now = dt.datetime.now()
        db_name = now.strftime("Reid_%Y%m%d.db")
        db_path = os.path.join(FLAGS.reid_db_path, db_name).replace("\\","/")
        reid_db = ImageDB(db_name=db_path)
        print("reid db_path: ", db_path)
        reid_db.delete_dbfile()
        reid_db.create_table()
        self.db_path = db_path
        #reid = Reid(db_path)

    # main process will run this database
    def save_db(self):
        renew_db = True
        while self.stop_main_ps.value:
            now = dt.datetime.now()
            t = now.timetuple()
            # t[6] consists of day name information. 0 = Monday. 4 = Friday.
            #print("t[6]: ", t[6])
            if t[6] == 2 or t[6] == 5: 
                if renew_db:
                    # stop human tracker processes
                    self.stop_tracker.value = 1
                    # create new database on Wednesday and Saturday, and only renew one time on each day.
                    self.create_new_db()
                    renew_db = False
                    print("New Database with timestamp [", now.strftime("%A, %d. %B %Y %I:%M%p"), ']')  
                    # reset human tracker processes
                    self.stop_tracker.value = 0 
                else:
                    time.sleep(1)
            else:
                renew_db = True
                #print("self.stop_tracker: ", self.stop_tracker.value)
                #print("main process ticks..")
                time.sleep(1)
        print("save_db loop is ended..")

    def signal_handler(self, sig, frame):
        print('Main Program: You pressed Ctrl+C!')

        # when ctrl + c, rename the database as interrupted 
        if os.path.isfile(self.db_path):
            now = dt.datetime.now()
            db_name = now.strftime("Reid_Interrupted_%Y%m%dT%H%M%S.db")
            interrupt_path = os.path.join(FLAGS.reid_db_path, "Interrupt", db_name).replace("\\","/")
            #os.rename(self.db_path, interrupt_path)
            shutil.move(self.db_path, interrupt_path)

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


def sequential_run(batch, cam, db_path, mps):
    mps.job.clear()
    #mps.new_job('database_ps', db_process, cam, FLAGS.parallel_ps, FLAGS.reid_db_path, FLAGS.merge_db_path)
    mps.cam = cam
    print("batch:", batch)
    gpu_num = 0
    for ch in batch:
        mps.new_job('camera_ch' + ch, camera_capture, FLAGS.online, int(ch), gpu_num, db_path)
        #gpu_num = 1 - gpu_num
    for j in mps.job:
        j.start()
    # main process in while loop, save database when cutoff date is reached
    mps.save_db()
    for j in mps.job:
        j.join()

def online_run(rtsp, cam, gpu, loc, db_path, mps):
    mps.job.clear()
    #mps.new_job('database_ps', db_process, cam, FLAGS.parallel_ps, FLAGS.reid_db_path, FLAGS.merge_db_path)
    mps.cam = cam
    for i in range(FLAGS.parallel_ps):
        # cam[i]:int , rtsp[i]:str
        mps.new_job('camera_ch' + str(cam[i]), camera_capture, FLAGS.online, cam[i], rtsp[i], gpu[i], loc[i], db_path)
        print("New online process for cam " + str(cam[i]))
    for j in mps.job:
        j.start()
    # main process in while loop, save database when cutoff date is reached
    mps.save_db()
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

    # create new database with timestamp
    mps.create_new_db()

    # initialize backup database
    #db_path = None
    # if FLAGS.db:
    #     db_path = "./database/Image_" + str(dt.datetime.now().strftime("%Y%m%dT%H%M%S")) + ".db"
    #     print("db_path main: ", db_path)
    #     img_db = ImageDB(db_path)
        #img_db.delete_dbfile()
        #img_db.create_table()

    # online mode
    if FLAGS.online:
        table = get_rtsp(FLAGS.rtsp_path)
        online_run(table.to_dict('dict')['rtsp'], table.to_dict('dict')['cam'], table.to_dict('dict')['gpu'], table.to_dict('dict')['loc'], mps.db_path, mps)
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
            table = get_rtsp(FLAGS.rtsp_path)
            # run new camera process
            for batch in ps_list:
                sequential_run(batch, table.to_dict('dict')['cam'], mps.db_path, mps)
        else:
            cam_stream(mps)
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
