
from .inference import reid_inference
from .utils.metrics import cosine_similarity, euclidean_distance
from .utils.to_sqlite import insert_vector_db, insert_human_db, insert_infer_db, load_gallery_from_db, convertToBinaryData, load_human_db, load_images_from_db
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
        self.reid.to_gallery_feat(img, flip= False) #always set FLIP = FALSE
        query_feat = self.reid.to_query_feat(img)
        self.reid.infer(query_feat, imgid, reranking=True) #always RERANKING = TRUE
        print("[Reid]: --- %s seconds ---" % (time.time() - start_time))