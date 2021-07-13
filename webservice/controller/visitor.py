from flask import Flask, jsonify, request


def visitor(app, path):
    db = app['db']
    app = app['app']
    @app.route(path, methods=["GET", "POST"])
    def visitor():
        if request.method == "POST":
            try:
                start_date = request.json['start_date']
                end_date = request.json['end_date']
                query = f"SELECT * FROM customer_data_total WHERE date BETWEEN '{start_date}' AND '{end_date}'"
                rows = db.query(query)
                resp = jsonify(rows)
                resp.status_code = 200
                return resp

            except Exception as e:
                raise e
            
        else:
            try:
                query = "SELECT * FROM customer_data_total"
                rows = db.query(query)
                resp = jsonify(rows)
                resp.status_code = 200
                return resp

            except Exception as e:
                raise e
