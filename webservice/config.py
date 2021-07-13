import os
from configparser import ConfigParser

from dotenv import load_dotenv

load_dotenv()

# Parsing app.ini.example.ini to app.ini
app = ConfigParser()
app.read("config/app.ini.example")
app["app"]["db"] = os.getenv("DB")
app["app"]["storage"] = os.getenv("STORAGE")
with open("config/app.ini", "w") as f:
    app.write(f)

db_section = app["app"]["db"]
storage_section = app["app"]["storage"]

# Parsing database.ini.example to database.ini
db = ConfigParser()
db.read("config/database.ini.example")
db[db_section]["host"] = os.getenv("DBHOST")
db[db_section]["port"] = os.getenv("DBPORT")
db[db_section]["user"] = os.getenv("DBUSER")
db[db_section]["password"] = os.getenv("DBPASSWORD")
db[db_section]["db"] = os.getenv("DBNAME")
with open("config/database.ini", "w") as f:
    db.write(f)
