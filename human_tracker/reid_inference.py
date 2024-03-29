# Add parent folder (root folder) into the sys.path, so that this module can import reid functions
# link1: https://stackoverflow.com/questions/36622183/how-to-import-a-function-from-parent-folder-in-python
# link2: https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time#answer-14132912
import os, sys
p = os.path.abspath('../reid')
if p not in sys.path:
    sys.path.append(p)
from inference import reid_inference
from utils.metrics import cosine_similarity, euclidean_distance
from utils.to_sqlite import insert_vector_db, insert_human_db, insert_infer_db, load_gallery_from_db, convertToBinaryData, load_human_db, load_images_from_db, convertBlobtoIMG
import utils.to_sqlite as sql
import numpy as np
import datetime as dt
import time

#init class
#reid = reid_inference()

#img_id, cam_id, img_list_all = load_images_from_db(db = 'Image_12Jul_P2.db', table = 'VectorKB_Table') 

#img = img_list_all[0]
#imgid = img_id[0]

class Reid():
    def __init__(self, db_path):
        # if you want to use general functions in reid_inference.py, insert the db_path into the parameter for class constructor. Call the function after the object is created(e.g. reid.to_gallery_feat(imgid, pil_img))
        self.reid = reid_inference(db_path)
        # self.db_path for member class.
        self.db_path = db_path

    def get_timestamp(self):
        timestamp = dt.datetime.now()
        time_filename = timestamp.strftime("%Y%m%dT%H%M%S")
        return timestamp, time_filename

    def get_imgid(self, cam_id, track_id):
        timestamp, time_filename = self.get_timestamp()
        img_id = str(cam_id) + '_' + str(track_id) + '_' + str(time_filename) 
        return img_id

    def run(self, imgid, img, loc):
        start_time = time.time()
        pil_img = convertBlobtoIMG(img)
        self.reid.to_gallery_feat(imgid, pil_img, loc) #always set FLIP = FALSE
        #query_feat = self.reid.to_query_feat(img)
        #self.reid.infer(query_feat, imgid, reranking=True) #always RERANKING = TRUE
        print("[Reid.to_gallery_feat]: --- %s seconds ---" % (time.time() - start_time))

    def run_reid(self):
        print("run_reid db_path: ", self.db_path)
        # if you want to use inner function inside to_sqlite.py (e.g. load_images_from_db()), insert the db_path in the sql.db_path. Remember to import utils.to_sqlite on top.
        sql.db_path = self.db_path 
        _, _, img_list_all = load_images_from_db()
        for img in img_list_all:
            start_time = time.time()
            #pil_img = convertBlobtoIMG(img)
            query_feat = self.reid.to_query_feat(img)
            print("query_feat: ", query_feat)
            result = self.reid.infer(query_feat, rerank_range = 50)
            print("--- %s seconds ---" % (time.time() - start_time))