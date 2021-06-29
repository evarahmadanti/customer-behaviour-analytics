from app import app
from flaskext.mysql import MySQL

mysql = MySQL()

#MySQL config
app.config["MYSQL_DATABASE_USER"] = "admin"
app.config["MYSQL_DATABASE_PASSWORD"] = "02_Jan99"
app.config["MYSQL_DATABASE_DB"] = "customer_behaviour_analytics"
app.config["MYSQL_DATABASE_HOST"] = "localhost"
mysql.init_app(app)