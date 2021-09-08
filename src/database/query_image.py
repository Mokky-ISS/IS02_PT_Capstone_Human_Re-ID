from database.database import DbQuery

class DbQuery(DbQuery):
    def __init__(self) -> None:
        super().__init__('database/Image.db')
        self.table = 'VectorKB_Table'

    def get_cam_id_options(self):
        query = f"SELECT DISTINCT(cam_id) FROM {self.table}"
        df = super().query_data(query)
        return [{'label': f'Camera {id}', 'value': id} for id in df['cam_id'].values if int(id) > 0]

    def get_images(self, cam_id=None, start_datetime=None, end_datetime=None):
        query = f"SELECT img_id, cam_id, timestamp, patch_img AS img FROM {self.table}"
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
            query += f" timestamp >= '{start_datetime}'"
        if end_datetime is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" timestamp <= '{end_datetime}'"
        query += " ORDER BY cam_id, timestamp"
        return super().query_data(query)
