import sqlite3
import io
import numpy as np
import pandas as pd


class DbQuery(object):

    def __init__(self, db_path, table_name=None):
        self.conn = None
        self.cursor = None
        self.db_path = db_path
        self.table_name = table_name
        self.data = None

    # decorator for sqlite connection
    def _con_sqlite(func):
        def inner(self, *args, **kwargs):
            #print("Run Sqlite.")
            self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES |
                                        sqlite3.PARSE_COLNAMES)
            self.cursor = self.conn.cursor()

            func(self, *args, **kwargs)

            #self.conn.commit()
            self.cursor.close()
            self.conn.close()
        return inner
    #_con_sqlite = staticmethod(_con_sqlite)

    @_con_sqlite
    def _sql_query(self, query):
        self.cursor.execute(query)
        column_names = [desc[0] for desc in self.cursor.description]
        self.data = pd.DataFrame(self.cursor.fetchall(), columns=column_names)

    def query_data(self, query):
        self._sql_query(query)
        return self.data
