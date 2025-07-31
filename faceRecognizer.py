import dlib
import numpy as np

class FaceRecognizer:
    def __init__(self, predictor_path, encoder_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_recognizer = dlib.face_recognition_model_v1(encoder_path)

    def detect_faces(self, image):
        return self.detector(image, 1)

    def get_face_descriptor(self, image, face):
        shape = self.predictor(image, face)
        return np.array(self.face_recognizer.compute_face_descriptor(image, shape))

    def recognize_faces(self, image):
        faces = self.detect_faces(image)
        descriptors = [self.get_face_descriptor(image, face) for face in faces]
        return descriptors
    
    def face_similarity(self, desc1, desc2):
        """ Calculate the Euclidean distance between two face descriptors. """
        return np.linalg.norm(desc1 - desc2)