from database.database import DbQuery


class DbQuery(DbQuery):
    def __init__(self, path_db) -> None:
        super().__init__(path_db)


    def get_images(self, img_id):
        #query = "SELECT h.img_id, h.human_id, h.inference_datetime, i.img FROM human_table AS h"
        #query += " INNER JOIN vectorkb_table AS v ON h.img_id=v.img_id"
        query = "SELECT img_id, cam_id, create_datetime AS timestamp, img FROM vectorkb_table AS v"
        #query += " INNER JOIN face_scores_table AS f ON h.img_id=f.img_id"
        if img_id is not None:
            if " WHERE " in query:
                query += " AND"
            else:
                query += " WHERE"
            query += f" img_id == '{img_id}'"

        query += " ORDER BY cam_id"
        #print(query)
        return super().query_data(query)
