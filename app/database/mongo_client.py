import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB", "waste_ai")

client = MongoClient(MONGODB_URI, uuidRepresentation="standard")
db = client[DB_NAME]
