from controller.elapsed_times import elapsed_time
from controller.favorite import favorite
from controller.predict_time import predict_time
from controller.predict_visitor import predict_visitor
from controller.time import busy_time
from controller.visitor import visitor

# from .time import time

def router(app):
    # db = app['db']
    visitor(app, "/api/visitor")
    busy_time(app, "/api/time")
    favorite(app, "/api/favorite")
    predict_visitor(app, "/api/predict_visitor")
    elapsed_time(app, "/api/elapsed_times")
    predict_time(app, "/api/predict_time")


