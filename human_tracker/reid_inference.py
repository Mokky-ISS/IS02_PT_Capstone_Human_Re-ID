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
import numpy as np
import time

#init class
#reid = reid_inference()

#img_id, cam_id, img_list_all = load_images_from_db(db = 'Image_12Jul_P2.db', table = 'VectorKB_Table') 

#img = img_list_all[0]
#imgid = img_id[0]

class Reid():
    def __init__(self):
        self.reid = reid_inference()     

    def run(self, imgid, img):
        start_time = time.time()
        pil_img = convertBlobtoIMG(img)
        self.reid.to_gallery_feat(imgid, pil_img) #always set FLIP = FALSE
        #query_feat = self.reid.to_query_feat(img)
        #self.reid.infer(query_feat, imgid, reranking=True) #always RERANKING = TRUE
        print("[Reid.to_gallery_feat]: --- %s seconds ---" % (time.time() - start_time))