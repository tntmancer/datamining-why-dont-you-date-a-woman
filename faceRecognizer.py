import dlib
import numpy as np

class FaceRecognizer:
    def __init__(self, predictor_path, encoder_path):
        """ Initialize the face recognizer with the given model paths. """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_recognizer = dlib.face_recognition_model_v1(encoder_path)

    def detect_faces(self, image):
        """ Detect faces in the given image. """
        return self.detector(image, 1)

    def get_face_descriptor(self, image, face):
        """ Get the face descriptor for a detected face. """
        shape = self.predictor(image, face)
        return np.array(self.face_recognizer.compute_face_descriptor(image, shape))

    def recognize_faces(self, image):
        """ Recognize faces in the given image and return their descriptors. """
        faces = self.detect_faces(image)
        descriptors = [self.get_face_descriptor(image, face) for face in faces]
        return descriptors
    
    def face_similarity(self, desc1, desc2):
        """ Calculate the Euclidean distance between two face descriptors. """
        return np.linalg.norm(desc1 - desc2)