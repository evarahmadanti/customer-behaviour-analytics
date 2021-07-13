import datetime
from datetime import timedelta

from flask import Flask, jsonify, request


def predict_visitor(app, path):
    db = app['db']
    app = app['app']
    @app.route(path, methods=["GET"])
    def predict_visitor():
        try:
            # date_now = datetime.datetime.now()
            # start_date = request.json[date_now]
            # end_date = date_now.timedelta(days=30)
            # end_date = request.json[end_date]
            query = f"SELECT * FROM customer_data_dummy WHERE date BETWEEN '2021-05-01' AND '2021-05-31'"
            rows = db.query(query)
            resp = jsonify(rows)
            resp.status_code = 200
            return resp

        except Exception as e:
            raise e
