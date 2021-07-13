import json
from datetime import datetime as dt

import pandas as pd
from flask import Flask, jsonify, request
from pandas.api.types import CategoricalDtype


def busy_time(app, path):
    db = app['db']
    app = app['app']

    @app.route(path, methods=["GET", "POST"])
    def busy_time():
        if request.method == "POST":
            try:
                start_date = request.json['start_date']
                end_date = request.json['end_date']
                query = f"SELECT * FROM customer_data_total WHERE date BETWEEN '{start_date}' AND '{end_date}'"
                conn = db.db_client.db_conn
                rows = pd.read_sql_query(query, con=conn)
                days_name = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                days_categorical = CategoricalDtype(categories=days_name, ordered=True)
                rows['day'] = rows['date'].dt.day_name()
                rows['day'] = rows['day'].astype(days_categorical)
                # rows['date'] = pd.to_datetime(rows['date'])
                df_rows = rows.groupby(rows['day']).sum()
                analytics_data = df_rows['total_visitor']
                to_json = analytics_data.to_json()
                parsed = json.loads(to_json)
                parsed = json.dumps(parsed, indent=4)
                # parsed.status_code = 200
                return parsed

            except Exception as e:
                raise e
            
        else:
            try:
                query = "SELECT * FROM customer_data_total"
                conn = db.db_client.db_conn
                rows = pd.read_sql_query(query, con=conn)
                days_name = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                days_categorical = CategoricalDtype(categories=days_name, ordered=True)
                rows['day'] = rows['date'].dt.day_name()
                rows['day'] = rows['day'].astype(days_categorical)
                # rows['date'] = pd.to_datetime(rows['date'])
                df_rows = rows.groupby(rows['day']).sum()
                analytics_data = df_rows['total_visitor']
                to_json = analytics_data.to_json()
                parsed = json.loads(to_json)
                parsed = json.dumps(parsed, indent=4)
                # parsed.status_code = 200
                return parsed

            except Exception as e:
                raise e
