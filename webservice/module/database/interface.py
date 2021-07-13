# interface for creating database client API

import abc


class DatabaseInterface(metaclass=abc.ABCMeta):
    """ Interface to create database client API """
    @classmethod
    def __subclasshook__(cls, subclass):
        return(hasattr(subclass, "__init__") and
               callable(subclass) and
               hasattr(subclass, "query") and
               callable(subclass.query) and
               hasattr(subclass, "close") and
               callable(subclass.close) or
               NotImplemented)

    @abc.abstractmethod
    def __init__(self, cfg: dict):
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, query: str, args=None, **kwargs):
        """ Query data """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """ Close database connection """
        raise NotImplementedError
