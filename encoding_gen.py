import os
import cv2
import face_recognition
import pickle


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
    with open("encodings.pickle", "wb") as f:
        pickle.dump([encodings, names], f)
    print("Done!") 