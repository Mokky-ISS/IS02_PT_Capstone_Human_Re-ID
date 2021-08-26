import torch
torch.cuda.set_device(1)
import os
import numpy as np
import cv2
from PIL import Image
import math
import matplotlib.pyplot as plt
from config import Config
from utils.to_sqlite import insert_vector_db, load_gallery_from_db, convertToBinaryData, convertImgtoBlob, convertBlobtoIMG
import utils.to_sqlite as sql
from utils.reranking import re_ranking

from model import make_model
from torch.backends import cudnn
import torchvision.transforms as T
from utils.metrics import cosine_similarity, euclidean_distance
import pickle


class reid_inference:
    """Reid Inference class.
    """

    def __init__(self, db_path):
        sql.db_path = db_path
        cudnn.benchmark = True
        self.Cfg = Config()
        self.model = make_model(self.Cfg, 255)
        self.model.load_param(self.Cfg.TEST_WEIGHT)
        self.model = self.model.to('cuda')
        self.transform = T.Compose([
                T.Resize(self.Cfg.INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        print(f'Model loaded with weight from {self.Cfg.TEST_WEIGHT}')
        self.model.eval()
        print('Ready to Eval')
        print('Loading from DB...')
        self.all_img_id, self.all_patch_img, self.all_gal_feat, self.all_cam_id = load_gallery_from_db() #load from vectorkb_table
        self._tmp_img = ""
        self._tmp_galfeat = ""
        print('Data loaded. You can start infer an image using to_gallery_feat --> query_feat --> infer')


    def to_gallery_feat(self, img_id, image_patch_or_path, flip=True, norm=True):
        """
        Use to build gallery feat on images picked from Deep Sort.
        This is different from normal query feature extraction as this has flipped & normed feature,
        to improve the matching precision.
        Takes image path or PIL image directly.
        To be combined with INFER function at the end of troubleshooting
        """
        if type(image_patch_or_path) is str:
            query_img = Image.open(image_patch_or_path)
        else:
            query_img = image_patch_or_path
        
        input = torch.unsqueeze(self.transform(query_img), 0)
        input = input.to('cuda')
        with torch.no_grad():
            if flip:
                gal_feat = torch.FloatTensor(input.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(input.size(3) - 1, -1, -1).long().cuda()
                        input = input.index_select(3, inv_idx)
                    f = self.model(input)
                    gal_feat = gal_feat + f
            else:
                gal_feat = self.model(input)

            if norm:
                gal_feat = torch.nn.functional.normalize(gal_feat, dim=1, p=2)

            query_img_blob = convertImgtoBlob(query_img)
            cam_id = img_id.split('_')[0]
            track_id = img_id.split('_')[1]
            insert_vector_db(img_id, query_img_blob, pickle.dumps(gal_feat), cam_id, track_id)
            self.all_img_id.append(img_id)
            self.all_patch_img.append(query_img)
            self.all_gal_feat = torch.cat([self.all_gal_feat, gal_feat])
            self.all_cam_id.append(cam_id)

            return gal_feat




    def to_query_feat(self, image_patch_or_path):
        """
        image - input image path.
        for finding query feature, no flipping and normalization is done.
        This function returns feature (1,2048) tensor.

        """
        if type(image_patch_or_path) is str:
            query_img = Image.open(image_patch_or_path)
        else:
            query_img = image_patch_or_path

        input = torch.unsqueeze(self.transform(query_img), 0)
        input = input.to('cuda')
        with torch.no_grad():
            query_feat = self.model(input)
        return query_feat




    def infer(self, query_feat, thres= 0.6, reranking = True, rerank_range = 50):

        #cosine
        try:
            dist_mat = torch.nn.functional.cosine_similarity(query_feat, self.all_gal_feat).cpu().numpy()
        except RuntimeError:
            print('Table is empty.')
            return []
        indices = np.argsort(dist_mat)[::-1][:rerank_range] #to test if use 50 or use all better
        
        if reranking:
            candidate_gal_feat = torch.index_select(self.all_gal_feat, 0, torch.tensor([indices]).cuda()[0])
            rerank_dist = re_ranking(query_feat, candidate_gal_feat, k1=30, k2=6, lambda_value=0.3)[0]
            rerank_idx = np.argsort(1-rerank_dist)[::-1]
            indices = np.array([indices[i] for i in rerank_idx])
            rerank_dist_ind = np.array([rerank_dist[i] for i in rerank_idx])

        all_result = []
        for idx, i in enumerate(indices):
            if dist_mat[i] < thres:
                continue
            
            tmp = dict()
            tmp['idx'] = i
            tmp['cam_id'] = self.all_cam_id[i]
            tmp['img_id'] = self.all_img_id[i]
            tmp['dist'] = dist_mat[i]
            tmp['rerank_dist'] = rerank_dist_ind[idx]
            tmp['ranking'] = idx
            

            all_result.append(tmp)
        return all_result






if __name__ == "__main__":
    reid = reid_inference()
    print(len(reid.all_img_id), len(reid.all_patch_img), len(reid.all_gal_feat), len(reid.all_cam_id))
    #print("Image in 10th row is", reid.all_img_id[10], "with cam ID =", reid.all_cam_id[10])

    img = './reid/example/0002_c3s1_UO5K20210612T1522373990_00.jpg'
    query_feat = reid.to_query_feat(img)
    result = reid.infer(query_feat, rerank_range = 50)

    print(result)
