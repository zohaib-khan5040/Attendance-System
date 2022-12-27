import base64
import numpy as np
from pymongo import MongoClient
from config import db_faces, db_logs

def str_to_arr(b_str):
    '''
    Decodes a binary string to a 1D numpy array
    '''
    return np.frombuffer(base64.b64decode(b_str), dtype=np.float64)

def arr_to_str(arr):
    '''
    Encodes a 1D numpy array to a binary string
    '''
    return base64.b64encode(arr).decode()

def load_database():
    known_encodings = []
    known_names = []
    for face in db_faces.find():
        encoding = str_to_arr(face["encoding"])
        known_encodings.append(encoding)
        known_names.append(face["name"])
    
    return known_encodings, known_names

if __name__=="__main__":
    known_encodings, known_names = load_database()
    print(known_encodings)
    print(known_names)