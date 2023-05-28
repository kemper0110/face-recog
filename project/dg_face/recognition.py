from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


def make_cnn_model(model='vgg16'):
    return VGGFace(include_top=False, model=model, input_shape=(224, 224, 3), pooling='avg')


def face_encodings(image, faces):
    pass
