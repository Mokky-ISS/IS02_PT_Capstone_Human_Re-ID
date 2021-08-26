import sqlite3
from sqlite3 import Error
import pickle
import torch
from PIL import Image
import io

# default path, will be changed by other script.
db_path = '../reid/database/reid_db.db'

def insert_vector_db(img_id, img, feat_vec, cam_id, track_id):
    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()
    
        Query = """
            INSERT INTO vectorkb_table (img_id, img, vector_tensor, cam_id, track_id) 
            VALUES (?,?,?,?,?)
            """

        array = (img_id, img, feat_vec, cam_id, track_id)
        Query = cursor.execute(Query, array)
        sqliteConnection.commit()

        cursor.close()
        sqliteConnection.close()
        
    except Error as err:
        print("Connection error to vector table", err)


def insert_human_db(img_id, human_id, inf_type):
    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()
    
        Query = """
            INSERT INTO human_table (img_id, human_id, type) 
            VALUES (?,?,?)
            """

        array = (img_id, human_id, inf_type)
        Query = cursor.execute(Query, array)
        sqliteConnection.commit()

        cursor.close()
        sqliteConnection.close()
        
    except Error as err:
        print("Connection error to human table", err)



def insert_infer_db(record):
    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()
    
        Query = """
            INSERT INTO inference_table (query_img_id, query_img, match_1_img_id, match_1_img, match_1_dist, match_2_img_id,match_2_img, match_2_dist, match_3_img_id, match_3_img, match_3_dist) 
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """

        Query = cursor.execute(Query, record)
        sqliteConnection.commit()

        cursor.close()
        sqliteConnection.close()
        
    except Error as err:
        print("Connection error to inference table", err)


def unpickle_blob(blob):
    if type(blob) is list:
        return torch.cat(list((pickle.loads(bob) for bob in blob)), dim=0)
        
    else:
        return pickle.loads(blob)


def convertBlobtoIMG(blob):
    # to convert blob to PIL image
    if type(blob) is list:
        return [Image.open(io.BytesIO(i)) for i in blob]
    else:
        return Image.open(io.BytesIO(blob))


def convertImgtoBlob(PIL):
    # to convert PIL image back to blob
    if type(PIL) is list:
        blob_list = []
        for each in PIL:
            stream = io.BytesIO()
            each.save(stream, format="JPEG")
            blob_list.append(stream.getvalue())
    else:
        stream = io.BytesIO()
        PIL.save(stream, format="JPEG")
        return stream.getvalue()


def convertToBinaryData(filename):
    # to convert image path to blob
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData



def load_gallery_from_db():
    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()

        cursor.execute("select img_id, img, vector_tensor, cam_id from vectorkb_table")
        result = cursor.fetchall()
        img_id =list(list(zip(*result))[0])
        patch_img = convertBlobtoIMG(list(list(zip(*result))[1]))
        gal_feat = unpickle_blob(list(list(zip(*result))[2]))
        cam_id =list(list(zip(*result))[3])

        cursor.close()
        sqliteConnection.close()
        return img_id, patch_img, gal_feat, cam_id

    except IndexError:
        #empty table
        return [],[],torch.tensor([]).cuda(),[]

    except Error as err:
        print("Connection error to Sql", err)


def load_human_db():
    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()

        cursor.execute("select img_id, human_id from human_table")
        result = cursor.fetchall()
        human_dict = dict(result)

        cursor.close()
        sqliteConnection.close()
        return human_dict

    except Error as err:
        print("Connection error to Sql", err)




#####################################
# -- EXTRACT IMAGES FROM DB (TEST)  #
#####################################



def load_images_from_db(table = 'vectorkb_table'):
    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()

        cursor.execute(f"select img_id, cam_id, img from {table}")
        result = cursor.fetchall()
        img_id  = list(list(zip(*result))[0])
        cam_id  = list(list(zip(*result))[1])
        img_list = convertBlobtoIMG(list(list(zip(*result))[2]))
        

        cursor.close()
        sqliteConnection.close()
        return img_id, cam_id, img_list

    except Error as err:
        print("Connection error to Sql", db_path, err)