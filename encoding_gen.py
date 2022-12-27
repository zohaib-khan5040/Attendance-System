import os
import cv2
import face_recognition
import pickle
from pymongo import MongoClient
from config import conn
from utils import str_to_arr, arr_to_str

# Set up the collections
db_faces = conn.AttendanceDemo.FacesDatabase
db_logs = conn.AttendanceDemo.Logs


def get_images_and_names(image_folder_path):

    names = []
    images = []

    for path in os.listdir(image_folder_path):
        # Append the name
        name = os.path.splitext(path)[0]
        names.append(name)

        # Read in the image
        img = cv2.imread(os.path.join(image_folder_path, path))
        images.append(img)

    return images, names

def find_encodings(images):

    encodings = []

    for img in images:
        # Make sure we're in RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get the encoding
        encoding = face_recognition.face_encodings(img)[0]
        encodings.append(encoding)
    
    return encodings

if __name__ == "__main__":
    print("Reading images and names...")
    images, names = get_images_and_names("faces")
    print("Done!")

    print("Finding encodings...")
    encodings = find_encodings(images)
    print("Done!")

    print("Saving encodings...")
    count = 0
    for encoding, name in zip(encodings, names):
        
        # Convert the encoding to a string
        encoding_str = arr_to_str(encoding)

        # Check if the name is already in the database
        if db_faces.find_one({"name": name}):
            # If it is, update the encoding
            db_faces.update_one(
                {"name": name}, 
                {"$set": {"encoding": encoding_str}}
            )
        else:
            # If it isn't, insert the name and encoding
            db_faces.insert_one(
                {"name": name, "encoding": encoding_str}
            )
        
        count += 1
    
    print(f"Finished: saved {count} encodings!")