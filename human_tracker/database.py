import sqlite3
import os
import sys
import io
import numpy as np
import datetime


class ImageDB(object):

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.db_path = "./database/Image.db"
        self.table_name = "VectorKB_Table"
        # change the default column title here
        self.col_titles = "(img_id INT, cam_id INT, timestamp TIMESTAMP, track_id INT, patch_img BLOB, patch_np array, patch_bbox array, frame_num INT)"
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

    def get_timestamp(self):
        timestamp = datetime.datetime.now()
        time_filename = datetime.datetime.now().strftime("%Y%m%dT%H%M")
        return timestamp, time_filename

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
    def insert_data(self, cam_id, track_id, patch_img, patch_np, patch_bbox, frame_num):
        timestamp, time_filename = self.get_timestamp()
        img_id = str(cam_id) + '_' + str(track_id) + '_' + str(time_filename)
        # change the input column data here
        col = ("img_id", "cam_id", "timestamp", "track_id", "patch_img", "patch_np", "patch_bbox", "frame_num")
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


def reid_table():
    img_db = ImageDB()
    # add new reid table here
    img_db.create_table(table_name="Inference_Table", col_titles="(img_id INT, cam_id INT, timestamp TIMESTAMP, track_id INT)")


# UNCOMMEND THE MAIN FUNCTIONS TO TEST THEM OUT
if __name__ == "__main__":
    # pass
    # list_data()
    # query_data()
    # get_image()
    reid_table()
