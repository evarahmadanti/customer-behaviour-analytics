from flask import Flask, jsonify, request


def favorite(app, path):
    db = app['db']
    app = app['app']

    @app.route(path, methods=["GET", "POST"])
    def favorite():
        if request.method == "POST":
            try:
                start_date = request.json['start_date']
                end_date = request.json['end_date']
                query = f"SELECT * FROM customer_data_total WHERE date BETWEEN '{start_date}' AND '{end_date}'"
                rows = db.query(query)
                resp = jsonify(rows)
                return resp

            except Exception as e:
                raise e
            
        else:
            try:
                query = "SELECT * FROM customer_data_total"
                rows = db.query(query)
                resp = jsonify(rows)
                return resp

            except Exception as e:
                raise e
