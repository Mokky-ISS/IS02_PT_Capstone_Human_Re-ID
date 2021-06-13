from database.database import DbQuery
import pandas as pd
import sqlite3


class DbQuery(DbQuery):
    def __init__(self) -> None:
        super().__init__('database/reid_db_labelling.db')
        self.table = "mislabelled_table"

    def get_mislabelled(self):
        try:
            query = f"SELECT m.img_id AS img_id, m.is_mislabelled AS is_mislabelled, h.human_id AS human_id FROM {self.table} AS m"
            query += " INNER JOIN human_table AS h ON h.img_id=m.img_id"
            df = super().query_data(query)
            df.is_mislabelled = df.is_mislabelled.astype('bool')
            return df
        except:
            return pd.DataFrame(columns=['img_id', 'is_mislabelled', 'human_id'])

    def save_mislabelled(self, df):
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES |
                               sqlite3.PARSE_COLNAMES) as conn:
            df.to_sql(self.table, con=conn, index=False, if_exists='replace')

    def reset_mislabelled(self):
        query = f"DROP TABLE {self.table}"
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
