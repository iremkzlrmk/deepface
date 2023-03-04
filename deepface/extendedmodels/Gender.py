from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Activation
from deepface.basemodels import VGGFace


labels = ["Woman", "Man"]


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
):

    model = VGGFace.baseModel()

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    gender_model.load_weights("deepface/weights/gender_model_weights.h5")

    return gender_model
