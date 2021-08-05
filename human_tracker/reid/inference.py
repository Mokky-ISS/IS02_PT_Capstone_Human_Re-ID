import torch
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from .config import Config
from .utils.to_sqlite import insert_vector_db, insert_human_db, insert_infer_db, load_gallery_from_db, convertToBinaryData, load_human_db, convertImgtoBlob, convertBlobtoIMG
from .utils.reranking import re_ranking

from .model import make_model
from torch.backends import cudnn
import torchvision.transforms as T
from .utils.metrics import cosine_similarity, euclidean_distance
import pickle


class reid_inference:
    """Reid Inference class.
    """

    def __init__(self):
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
        self.all_img_id, self.all_patch_img, self.all_gal_feat = load_gallery_from_db() #load from vectorkb_table
        self.human_dict = load_human_db()
        self._tmp_img = ""
        self._tmp_galfeat = ""
        print('Data loaded. You can start infer an image using to_gallery_feat --> query_feat --> infer')
    


    def to_gallery_feat(self, image_patch_or_path, flip=True, norm=True):
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

            self._tmp_img = query_img  #temp save PIL image here
            self._tmp_galfeat = gal_feat #temp save gal_feat here
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



    def infer(self, query_feat, query_img_id, top_k= 3, reranking=True):
        if len(self.all_gal_feat)>0:
            # if reranking:
            #     dist_mat = 1 - re_ranking(query_feat, self.all_gal_feat, k1=30, k2=10, lambda_value=0.2)[0]
            #     indices = np.argsort(dist_mat)[::-1]

            # else:
            dist_mat = torch.nn.functional.cosine_similarity(query_feat, self.all_gal_feat).cpu().numpy()
            indices = np.argsort(dist_mat)[::-1][:50] #to test if use 50 or use all better

            #do reranking
            if reranking:
                candidate_gal_feat = torch.index_select(self.all_gal_feat, 0, torch.tensor([indices]).cuda()[0])
                rerank_dist = re_ranking(query_feat, candidate_gal_feat, k1=30, k2=6, lambda_value=0.3)[0]
                rerank_idx = np.argsort(1-rerank_dist)[::-1]
                indices = np.array([indices[i] for i in rerank_idx])
            
            #if match found --> insert to human_table, need a human list too. make it into class
            #if no match found --> insert new identity to human_table.
            if dist_mat[indices[0]] >= self.Cfg.THRESHOLD:

                
                #match found
                matched_img_id = self.all_img_id[indices[0]]
                identity = self.human_dict[matched_img_id]
                print(f"Match found! Identity is {identity}")

                #insert to human_table & dict
                insert_human_db(query_img_id, identity, "Matched")
                self.human_dict[query_img_id] = identity

                #insert query image to gallery table & list. Recalling we have self.tmp variables
                query_img_blob = convertImgtoBlob(self._tmp_img)
                insert_vector_db(query_img_id, query_img_blob, pickle.dumps(self._tmp_galfeat) ) 

                #insert to record table
                record = [query_img_id , query_img_blob]
                for k in range(top_k):
                    try:
                        record.append(self.all_img_id[indices[k]])
                        record.append(convertImgtoBlob(self.all_patch_img[indices[k]]))
                        record.append(dist_mat.item(indices[k]))
                    except:
                        record.append(None)
                        record.append(None)
                        record.append(None)
                insert_infer_db(record)


            elif (len(indices)>= 2) and (dist_mat[indices[1]] >= 0.75):
                #match found
                matched_img_id = self.all_img_id[indices[1]]
                identity = self.human_dict[matched_img_id]
                print(f"Match found! Identity is {identity} --> SECOND MATCH")

                #insert to human_table & dict
                insert_human_db(query_img_id, identity, "Matched")
                self.human_dict[query_img_id] = identity

                #insert query image to gallery table & list. Recalling we have self.tmp variables
                query_img_blob = convertImgtoBlob(self._tmp_img)
                insert_vector_db(query_img_id, query_img_blob, pickle.dumps(self._tmp_galfeat) ) 

                #insert to record table
                record = [query_img_id , query_img_blob]
                for k in range(top_k):
                    try:
                        record.append(self.all_img_id[indices[k+1]])
                        record.append(convertImgtoBlob(self.all_patch_img[indices[k+1]]))
                        record.append(dist_mat.item(indices[k+1]))
                    except:
                        record.append(None)
                        record.append(None)
                        record.append(None)
                insert_infer_db(record)

            else:
                #no match found
                new_identity = str(max(map(int, self.human_dict.values()))+1)
                print(f"No match found! Creating new identity -- {new_identity}")

                #insert to human_table & dict
                insert_human_db(query_img_id, new_identity, "New")
                self.human_dict[query_img_id] = new_identity

                #insert query image to gallery table & list. Recalling we have self.tmp variables
                query_img_blob = convertImgtoBlob(self._tmp_img)
                insert_vector_db(query_img_id, query_img_blob, pickle.dumps(self._tmp_galfeat) ) 

                #insert to record table
                record = [query_img_id , query_img_blob]
                for k in range(top_k):
                    record.append(None)
                    try:
                        record.append(convertImgtoBlob(self.all_patch_img[indices[k]]))
                        record.append(dist_mat.item(indices[k]))
                    except:
                        record.append(None)
                        record.append(None)
                insert_infer_db(record)

            #Putting these records into memory database
            self.all_img_id.append(query_img_id)
            self.all_patch_img.append(self._tmp_img)
            self.all_gal_feat = torch.cat([self.all_gal_feat, self._tmp_galfeat])

                        
        else:
            #new record
            new_identity = str(1)
            print(f"No match found! Creating new identity -- {new_identity}")

            #insert to human_table & dict
            insert_human_db(query_img_id, new_identity, "New")
            self.human_dict[query_img_id] = new_identity

            #insert query image to gallery table & list. Recalling we have self.tmp variables
            query_img_blob = convertImgtoBlob(self._tmp_img)
            insert_vector_db(query_img_id, query_img_blob, pickle.dumps(self._tmp_galfeat) ) 

            #insert to record table
            record = [query_img_id , query_img_blob]
            for k in range(top_k):
                record.append(None)
                record.append(None)
                record.append(None)
            insert_infer_db(record)

            #Putting these records into memory database
            self.all_img_id.append(query_img_id)
            self.all_patch_img.append(self._tmp_img)
            self.all_gal_feat = torch.cat([self.all_gal_feat, self._tmp_galfeat])







    
    # def build_all_gallery(dir_to_gal_folder = self.Cfg.GALLERY_DIR, to_db = False):
    #     """
    #     TAKE NOTEE!! TO BE MODIFIED AS WE NO LONGER NEED TO MASS UPLOAD FROM
    #     IMG FILE TO GALLERY DB.
    #     """
    #     all_gal_feat = []
    #     all_img_id = os.listdir(dir_to_gal_folder) #this is id rather than path

    #     db_feat = []
    #     db_img = []

    #     print(f'Building gallery from {dir_to_gal_folder}...')
    #     for img in all_img_id:
    #         gal_feat = to_gallery_feat(dir_to_gal_folder + "/" + img)
    #         all_gal_feat.append(gal_feat)
    #         db_feat.append(pickle.dumps(gal_feat))
    #         db_img.append(convertToBinaryData(dir_to_gal_folder + "/" + img))

    #     all_gal_feat = torch.cat(all_gal_feat, dim=0)

    #     if to_db:
    #         db_img_path = [dir_to_gal_folder + "/" + img for img in all_img_id]
    #         db_humam_id = [img.split('_')[0] for img in all_img_id]
    #         insert_vector_db(all_img_id, db_img_path, db_img, db_feat)
    #         insert_human_db(all_img_id, db_humam_id)
    #         print('All gallery uploaded to DB.')
    #     else:
    #         return all_gal_feat, all_img_id

    # UNSUPPORTED
                    # plt.subplot(1, top_k+2, 1)
                    # plt.title('Query')
                    # plt.axis('off')
                    # query_img = Image.open(query_img_path)
                    # plt.imshow(np.asarray(query_img))

                    # for k in range(top_k):
                    #     plt.subplot(1, top_k+2, k+3)
                    #     name = str(indices[k]) + '\n' + '{0:.2f}'.format(dist_mat[indices[k]])
                    #     img = np.asarray(Image.open(self.all_img_path[indices[k]]))
                    #     plt.title(name)
                    #     plt.axis('off')
                    #     plt.imshow(img)
                    # plt.show()