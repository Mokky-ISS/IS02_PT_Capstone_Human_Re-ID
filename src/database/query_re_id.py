from src.database.database import DbQuery


class DbQuery(DbQuery):
    def __init__(self) -> None:
        super().__init__('database/reid_db.db')

    def get_human_id_options(self):
        query = "SELECT DISTINCT(human_id) FROM human_table"
        df_ids = super().query_data(query)
        return [{'label': id, 'value': id} for id in df_ids['human_id'].values if int(id) > 0]

    def get_images(self, human_id=None, start_datetime=None, end_datetime=None):
        query = "SELECT h.img_id, h.human_id, h.inference_datetime, v.img FROM human_table AS h"
        query += " INNER JOIN vectorkb_table AS v ON h.img_id=v.img_id"
        if human_id is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" h.human_id == '{human_id}'"
        if start_datetime is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" h.inference_datetime >= '{start_datetime}'"
        if end_datetime is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" h.inference_datetime <= '{end_datetime}'"
        query += " ORDER BY h.inference_datetime"
        #print(query)
        return super().query_data(query)
