import sqlite3
import os
import sys
import io
import numpy as np
import datetime


class ImageDB(object):
    main_db = "./database/Image.db"
    def __init__(self, db_name=main_db):
        self.conn = None
        self.cursor = None
        self.db_path = db_name
        #self.db_path = "./database/Image_complete.db"
        self.table_name = "vectorkb_table"
        # change the default column title here
        #self.col_titles = "(img_id INT, cam_id INT, timestamp TIMESTAMP, track_id INT, patch_img BLOB, patch_np array, patch_bbox array, frame_num INT, img_res_width INT, img_res_height INT)"
        # Important: Make sure the created column name in new table is same as the selected column name during selection by load_gallery_from_db from reid/utils/to_sqlite.py!
        self.col_titles = "(img_id INT, img BLOB, vector_tensor array, create_datetime TIMESTAMP, cam_id INT, track_id INT)"
        self.data = None
        self.image_name = []
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convert_array)

    # decorator for sqlite connection
    def _con_sqlite(func):
        def inner(self, *args, **kwargs):
            print("Run Sqlite.")
            self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES |
                                        sqlite3.PARSE_COLNAMES)
            self.cursor = self.conn.cursor()

            func(self, *args, **kwargs)

            self.conn.commit()
            self.cursor.close()
            self.conn.close()
        return inner
    #_con_sqlite = staticmethod(_con_sqlite)

    def delete_dbfile(self):
        if os.path.isfile(self.db_path):
            os.remove(self.db_path)

    def get_timestamp(self):
        timestamp = datetime.datetime.now()
        time_filename = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return timestamp, time_filename

    def get_imgid(self, cam_id, track_id):
        timestamp, time_filename = self.get_timestamp()
        img_id = str(cam_id) + '_' + str(track_id) + '_' + str(time_filename) 
        return img_id

    def load_directory(self, path='.'):
        for x in os.listdir(path):
            self.image_name.append(x)
        return self.image_name

    def adapt_array(self, arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    @_con_sqlite
    def create_table(self, table_name=None, col_titles=None):
        # https://stackoverflow.com/questions/1802971/nameerror-name-self-is-not-defined
        if table_name is None:
            table_name = self.table_name
        if col_titles is None:
            col_titles = self.col_titles

        command = ' '.join(("CREATE TABLE IF NOT EXISTS", table_name, col_titles))
        print(command)
        self.cursor.execute(command)

    @_con_sqlite
    def insert_data_old(self, cam_id, track_id, patch_img, patch_np, patch_bbox, frame_num, img_res_width, img_res_height):
        timestamp, time_filename = self.get_timestamp()
        img_id = str(cam_id) + '_' + str(track_id) + '_' + str(time_filename) + '_' + str(frame_num)
        # change the input column data here
        col = ("img_id", "cam_id", "timestamp", "track_id", "patch_img", "patch_np", "patch_bbox", "frame_num", "img_res_width", "img_res_height")
        col_str = '('+', '.join(col) + ')'
        command = ' '.join(("INSERT INTO", self.table_name, col_str, "VALUES (", ','.join(('?'*len(col))), ")"))
        print(command, col_str)
        # eval converts string tuple to tuple e.g. "(img_id,cam_id)" => (img_id, cam_id)
        self.cursor.execute(command, eval(col_str))

    @_con_sqlite
    def insert_data(self, cam_id, track_id, patch_img, patch_np):
        timestamp, time_filename = self.get_timestamp()
        img_id = self.get_imgid(cam_id, track_id)
        # change the input column data here
        col = ("img_id", "patch_img", "patch_np", "timestamp", "cam_id", "track_id")
        col_str = '('+', '.join(col) + ')'
        command = ' '.join(("INSERT INTO", self.table_name, col_str, "VALUES (", ','.join(('?'*len(col))), ")"))
        print(command, col_str)
        # eval converts string tuple to tuple e.g. "(img_id,cam_id)" => (img_id, cam_id)
        self.cursor.execute(command, eval(col_str))

    @_con_sqlite
    def fetch_data(self, *args):
        if len(args) == 0:
            col = "*"
        else:
            col = ', '.join(args)
        command = ' '.join(("SELECT", col, "FROM", self.table_name))
        print(command)
        self.cursor.execute(command)
        self.data = self.cursor.fetchall()

    # link: https://stackoverflow.com/questions/11653267/merge-tables-from-two-different-databases-sqlite3-python
    @_con_sqlite  # only use this method in main database!
    def merge_data(self, db_lists):
        for db in db_lists:
            db_cam = sqlite3.connect(db, detect_types=sqlite3.PARSE_DECLTYPES |
                                        sqlite3.PARSE_COLNAMES)
            db_cursor = db_cam.cursor()
            command = 'SELECT * FROM ' + self.table_name
            db_cursor.execute(command)
            output = db_cursor.fetchall()   # Returns the results as a list.
            # Insert those contents into another table.
            for row in output:
                self.cursor.execute('INSERT INTO ' + self.table_name + ' VALUES (?, ?, ?, ?, ?, ?)', row)
            # Cleanup
            db_cursor.close()

    # SAMPLE METHOD JUST FOR REFERENCE, DONT RUN THIS METHOD!
    # @_con_sqlite
    # def sample_method(self, cam_id, track_id, patch_img, patch_np, patch_bbox, frame_num):
    #     # create table this way
    #     self.cursor.execute("""
    #     CREATE TABLE IF NOT EXISTS example_table
    #     (data1_id INT, data2_id TIMESTAMP, data3_id BLOB, data4_id array)""")

    #     # insert table this way
    #     self.cursor.execute(""" INSERT INTO example_table
    #     (data1_id, data2_id, data3_id, data4_id) VALUES (?,?,?,?)""", (data1_id, data2_id, data3_id, data4_id))


def list_data():
    img_db = ImageDB()
    img_db.fetch_data()
    row = img_db.data[0]
    print("img_id:", row[0], type(row[0]))
    print("cam_id:", row[1], type(row[1]))
    print("timestamp:", str(row[2]), type(row[2]))
    print("track_id:", row[3], type(row[3]))
    #print("patch_img:", row[4])
    print("patch_np:", type(row[5]))
    print("patch_bbox:", row[6], type(row[6]))
    print("frame_num:", row[7], type(row[7]))


def query_data():
    img_db = ImageDB()
    img_db.create_table()
    img_db.fetch_data("img_id", "cam_id")
    # use img_db.data to get data at row 0 for cam_id
    row = img_db.data[0]
    print("cam_id:", row[1])


def get_image():
    img_db = ImageDB()
    img_db.fetch_data("patch_img", "img_id")
    row = img_db.data[0]
    with open("./database/{}.jpg".format(row[1]), "wb") as f:
        f.write(row[0])


def get_patch_np(id_num):
    img_db = ImageDB()
    img_db.fetch_data("patch_np", "img_id")
    row = img_db.data[id_num-1]
    return row[0], row[1]


def reid_table():
    img_db = ImageDB()
    # add new reid table here
    img_db.create_table(table_name="Inference_Table", col_titles="(img_id INT, cam_id INT, timestamp TIMESTAMP, track_id INT)")

# check for blur image


def list_blur_data():
    img_db = ImageDB()
    img_db.fetch_data()
    # blur images
    for i in []:
        row = img_db.data[i-1]
        print("img_id:", row[0], type(row[0]))
        print("patch_bbox:", row[6], type(row[6]))
        print("\n")
        with open("./database/img/{}.jpg".format(row[0]), "wb") as f:
            f.write(row[4])
    print("======================================")
    # non blur images
    for i in [88, 91, 108, 124, 125, 291, 358, 360, 534, 543, 632, 754, 773, 811, 828, 852, 916, 917, 997, 1153, 1287, 1292, 1487, 1532, 1533, 1546, 1561, 1563, 1654, 1663, 1664, 1773, 1790, 1800, 1815, 1854, 1898]:
        row = img_db.data[i-1]
        print("img_id:", row[0], type(row[0]))
        print("patch_bbox:", row[6], type(row[6]))
        print("\n")
        with open("./database/img/{}.jpg".format(row[0]), "wb") as f:
            f.write(row[4])


# UNCOMMEND THE MAIN FUNCTIONS TO TEST THEM OUT
if __name__ == "__main__":
    # pass
    list_blur_data()
    # query_data()
    # get_image()
    # reid_table()
