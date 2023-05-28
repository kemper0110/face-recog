import pathlib

import dlib

sp = dlib.shape_predictor(str(pathlib.Path.cwd().parent / 'dg_face/shape_predictor_5_face_landmarks.dat'))


def aligned_faces(img, faces):
    with_landmarks = dlib.full_object_detections()
    for detection in faces:
        with_landmarks.append(sp(img, detection))
    return dlib.get_face_chips(img, with_landmarks, size=224)


def aligned_face(img, face):
    return dlib.get_face_chip(img, sp(img, face), size=224)
