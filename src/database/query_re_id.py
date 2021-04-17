from src.database.database import DbQuery


class DbQuery(DbQuery):
    def __init__(self) -> None:
        super().__init__('database/reid_db.db')

    def get_human_id_options(self):
        df_ids = super().query_data('SELECT DISTINCT(human_id) FROM human_table')
        return [{'label': id, 'value': id} for id in df_ids['human_id'].values]
