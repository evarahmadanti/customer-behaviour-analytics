from .interface import DatabaseInterface
from .mysql import MySQLDB


class Database(DatabaseInterface):
    def __init__(self, db:str, cfg:dict):
        if db == "mysql":
            print("masuk Database()")
            self.db_client = MySQLDB(cfg)

    def query(self, query:str, args=None, **kwargs):
        """ Query data """
        return self.db_client.query(query, args)

    def close(self):
        """ Close database connection """
        self.db_client.close()
