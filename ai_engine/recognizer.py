import cv2
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", threshold=60):
        self.orb = cv2.ORB_create()
        self.threshold = threshold
        self.known_descriptors = []
        self.known_names = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory):
        for name in os.listdir(directory):
            person_dir = os.path.join(directory, name)
            if not os.path.isdir(person_dir):
                continue

            for file in os.listdir(person_dir):
                path = os.path.join(person_dir, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    self.known_descriptors.append(des)
                    self.known_names.append(name)

    def recognize(self, image, faces):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = []

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            kp, des = self.orb.detectAndCompute(face, None)

            name = "Unknown"
            if des is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                min_dist = float("inf")

                for i, known_des in enumerate(self.known_descriptors):
                    matches = bf.match(des, known_des)
                    if matches:
                        dist = np.mean([m.distance for m in matches])
                        if dist < min_dist:
                            min_dist = dist
                            name = self.known_names[i]

                if min_dist > self.threshold:
                    name = "Unknown"

            results.append(name)

        return results
