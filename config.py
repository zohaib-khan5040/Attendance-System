from pymongo import MongoClient

MONGO_STRING = "mongodb://localhost:27017" # "mongodb://localhost:27017" if running locally
conn = MongoClient(MONGO_STRING)

db_faces = conn.AttendanceDemo.FacesDatabase
db_logs = conn.AttendanceDemo.Logs


FRAME_FREQUENCY = 10 # How often to check for faces in a video stream