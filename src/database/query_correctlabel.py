from database.database import DbQuery
import pandas as pd
import sqlite3


class DbQuery(DbQuery):
    def __init__(self) -> None:
        super().__init__('database/reid_db_labelling.db')
        self.table = "correctlabel_table"

    def get_correctlabel(self):
        try:
            query = f"SELECT m.img_id AS img_id, m.is_correct AS is_correct, h.human_id AS human_id FROM {self.table} AS m"
            query += " INNER JOIN human_table AS h ON h.img_id=m.img_id"
            df = super().query_data(query)
            df.is_correct = df.is_correct.astype('bool')
            return df
        except:
            return pd.DataFrame(columns=['img_id', 'is_correct', 'human_id'])

    def save_correctlabel(self, df):
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES |
                               sqlite3.PARSE_COLNAMES) as conn:
            df.to_sql(self.table, con=conn, index=False, if_exists='replace')

    def reset_correctlabel(self):
        query = f"DROP TABLE {self.table}"
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
