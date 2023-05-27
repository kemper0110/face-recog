import dlib

detector = dlib.get_frontal_face_detector()


def detect_faces(img):
    return detector(img, 1)
