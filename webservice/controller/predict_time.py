import json
from datetime import datetime as dt

import pandas as pd
from flask import Flask, jsonify, request
from pandas.api.types import CategoricalDtype


def predict_time(app, path):
    db = app['db']
    app = app['app']

    @app.route(path, methods=["GET", "POST"])
    def predict_time():
        try:
            query = "SELECT * FROM customer_data_dummy WHERE date BETWEEN '2021-05-01' AND '2021-05-31'"
            conn = db.db_client.db_conn
            rows = pd.read_sql_query(query, con=conn)
            # days_categorical = CategoricalDtype(categories=weeks, ordered=True)
            rows.drop(columns=["id"], inplace=True)
            rows_sort = rows.resample('W-Mon', on='date')
            id_max = rows_sort['total_visitor'].idxmax()
            rows_max = rows.iloc[id_max]
            rows_max['date'] = rows_max['date'].dt.day_name()
            if len(rows_max) == 5:
                rows_max['weeks'] = [ '1st_Week', '2nd_Week', '3rd_Week', '4th_Week', '5th_Week']
            elif len(rows_max) == 6:
                rows_max['weeks'] = [ '1st_Week', '2nd_Week', '3rd_Week', '4th_Week', '5th_Week', '6th_Week']
            else:
                rows_max['weeks'] = [ '1st_Week', '2nd_Week', '3rd_Week', '4th_Week']
            to_json = rows_max.to_json(orient='records')
            return to_json

        except Exception as e:
            raise e
