from reid_inference import Reid
import time

db_path = '../reid/archive/Reid_20210826.db'
reid = Reid(db_path)
while True:
    reid.run_reid()
    time.sleep(1)