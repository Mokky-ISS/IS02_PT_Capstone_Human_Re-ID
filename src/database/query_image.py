from src.database.database import DbQuery

class DbQuery(DbQuery):
    def __init__(self) -> None:
        super().__init__('database/Image.db')
