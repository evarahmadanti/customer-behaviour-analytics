import re

import mysql.connector

from .interface import DatabaseInterface


class MySQLDB(DatabaseInterface):
    """ MySQL database API """
    def __init__(self, cfg):
        self.db_conn = mysql.connector.connect(**cfg)
        self.db_client = self.db_conn.cursor(dictionary=True)

    def query(self, query: str, args=None, **kwargs):
        """ Query data """
        select = query.split(' ')[0]
        if re.fullmatch("[Ss][Ee][Ll][Ee][Cc][Tt]", select):
            self.db_client.execute(query)
            return self.db_client.fetchall()
        try:
            self.db_client.execute(query)
            return self.db_conn.commit()
        except Exception as e:
            self.db_conn.rollback()
            raise e

    def close(self):
        """ Close MySQL connection """
        self.db_client.close()
        self.db_conn.close()
