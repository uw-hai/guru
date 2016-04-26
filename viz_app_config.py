import os

class Config(object):
    MONGO_HOST = os.environ['MONGO_HOST']
    MONGO_PORT = int(os.environ['MONGO_PORT'])
    MONGO_USER = os.environ.get('MONGO_USER', None)
    MONGO_PASS = os.environ.get('MONGO_PASS', None)
    MONGO_DBNAME = os.environ['MONGO_DBNAME']
    if 'STATIC_FOLDER' in os.environ:
        STATIC_FOLDER = os.environ['STATIC_FOLDER']
    else:
        STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
