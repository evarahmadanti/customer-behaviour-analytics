from types import CodeType
import pymysql
from app import app
from api import mysql
from flask import Flask, request
from flask import jsonify
from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/customer_data')
def customer_data():
    try:
        conn = mysql.connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("SELECT * FROM customer_data_detail")
        rows = cur.fetchall()
        resp = jsonify(rows)
        resp.status_code=200
        return resp

    except Exception as e:
        print(e)

    finally:
        cur.close()
        conn.close()
    
@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
        }
    resp = jsonify(message)
    resp.status_code = 404

    return resp

if __name__ == "__main__":
    app.run(debug=True)
        