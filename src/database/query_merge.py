from database.database import DbQuery
import pandas as pd
import sqlite3


class MergeDbQuery():
    def __init__(self) -> None:
        #dbPaths = [
        #    "database/reid_db_labelling (341-479)DC.db",
        #    "database/reid_db_(480-682)_YS.db"]
        dbPaths = ["database/reid_db_labelling.db"]
        self.correctlabel_table = "correctlabel_table"
        self.dbQueries = []
        for path in dbPaths:
            dbQuery = DbQuery(path)
            self.dbQueries.append(dbQuery)
        self.db_path = 'database/reid_db_merging.db'
        self.dbQuery = DbQuery(self.db_path)
        self.table_name = "merge_table"

    def get_all_correctlabel(self):
        try:
            query = f"SELECT m.img_id AS img_id, m.is_correct AS is_correct, h.human_id AS human_id, i.query_img AS img FROM {self.correctlabel_table} AS m"
            query += " INNER JOIN human_table AS h ON h.img_id=m.img_id"
            query += " INNER JOIN inference_table AS i ON h.img_id=i.query_img_id"
            query += " where m.is_correct == 1"
            df_list = []
            for dbQuery in self.dbQueries:
                df = dbQuery.query_data(query)
                df.is_correct = df.is_correct.astype('bool')
                df_list.append(df)
            df_merge = pd.concat(df_list, ignore_index=True)
            df_merge.sort_values('img_id', ignore_index=True)
            return df_merge
        except:
            return pd.DataFrame(columns=['img_id', 'is_correct', 'human_id'])

    def save_merge(self, listMerge):
        #print(listMerge)
        df = pd.DataFrame({'merges': [','.join(item) for item in listMerge]})
        #print(df)
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES |
                        sqlite3.PARSE_COLNAMES) as conn:
            df.to_sql(self.table_name, con=conn,
                      index=False, if_exists='replace')

    def get_merge(self):
        try:
            query = f"SELECT * FROM {self.table_name}"
            df = self.dbQuery.query_data(query)
            print(df)
            return [row[0].split(',') for _, row in df.iterrows()]
        except:
            return []

    def reset_merge(self):
        query = f"DROP TABLE {self.table_name}"
        #print(query)
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
