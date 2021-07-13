import os
import sys

from flask import Flask
from flask_cors import CORS

from controller.main import router
from module.database.main import Database
from module.helper.main import ConfigHelper

DEBUG_MODE = True if len(sys.argv) > 1 and sys.argv[1] == '-d' else False

def main():
    # get app config from config/app.ini
    app_cfg = ConfigHelper.get_file_config('config/app.ini', 'app')

    # get db config from config/database.ini
    db_engine = app_cfg['db']
    db_cfg = ConfigHelper.get_file_config('config/database.ini', db_engine)
    db = Database(db_engine, db_cfg)

    # configure flask and cors
    app = {'app': Flask(__name__), 'db': db}
    cors = CORS(app['app'], resources={r'/api/*': {'origins': '*'}})

    # run flask
    router(app)
    app['app'].run(host='0.0.0.0', port=os.getenv('PORT', 5500), debug=DEBUG_MODE)

if __name__ == "__main__":
    main()
