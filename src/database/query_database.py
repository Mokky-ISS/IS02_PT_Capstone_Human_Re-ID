from database.database import DbQuery
import pandas as pd

class DbQuery(DbQuery):
    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.table = 'VectorKB_Table'

    def get_table_list(self):
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        df = super().query_data(query)
        return df.name.tolist()

    def get_cam_id_options(self):
        query = f"SELECT * FROM {self.table} LIMIT 1"
        df = super().query_data(query)
        if "loc" in df.columns:
            query = f"SELECT DISTINCT cam_id, loc FROM {self.table} ORDER BY cam_id"
        else:
            query = f"SELECT DISTINCT cam_id FROM {self.table} ORDER BY cam_id"
        df = super().query_data(query)

        if "loc" in df.columns:
            return [{'label': f'Camera {row.cam_id} {row["loc"]}', 'value': row.cam_id} for _, row in df.iterrows() if row.cam_id > 0]
        else:
            return [{'label': f'Camera {row.cam_id}', 'value': row.cam_id} for _, row in df.iterrows() if row.cam_id > 0]

    def get_images(self, cam_id=None, start_datetime=None, end_datetime=None, img_id=None):
        query = f"SELECT * FROM {self.table} LIMIT 1"
        df = super().query_data(query)
        if "loc" in df.columns:
            query = f"SELECT img_id, cam_id, loc, create_datetime AS timestamp, img AS img FROM {self.table}"
        else:
            query = f"SELECT img_id, cam_id, create_datetime AS timestamp, img AS img FROM {self.table}"
        if cam_id is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" cam_id == '{cam_id}'"
        if start_datetime is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" (timestamp >= '{start_datetime}' OR timestamp = 'None')"
        if end_datetime is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" (timestamp < '{end_datetime}' OR timestamp = 'None')"
        if img_id is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" img_id == '{img_id}'"
        query += " ORDER BY cam_id, timestamp"
        return super().query_data(query)

    def get_date_range(self):
        query = f"SELECT MIN(create_datetime) AS min_date, MAX(create_datetime) AS max_date FROM {self.table}"
        df = super().query_data(query)
        df.min_date = pd.to_datetime(df.min_date)
        df.max_date = pd.to_datetime(df.max_date)
        return df.iloc[0].min_date, df.iloc[0].max_date
