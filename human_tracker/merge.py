from database import ImageDB
import time
db_path = "../reid/archive/Interrupt/Reid_20210906.db"
db_list = ["../reid/archive/Interrupt/Reid_20210908.db"]
img_db = ImageDB(db_path)

def file_reached():
    # tcp get db
    if received:
        return True
    else:
        return False

while True:
    if (file_reached()):
        img_db.merge_data(db_list)
    time.sleep(1)



