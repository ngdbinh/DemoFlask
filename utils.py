import hyper as hp
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import json


def _load_model():
    model_name = 'inception_resnet_v2'
    image_size = 256
    classes = 13
    model_path = 'saved'
    # Khởi tạo model
    base_model = InceptionResNetV2(include_top=False, weights=None,
                                   input_shape=(image_size, image_size, 3))

    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(units=classes, activation="sigmoid")(x)

    model = tf.keras.Model(base_model.input, output)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(os.path.join(model_path, f'model.{model_name}.h5'))
    return model


def _preprocess_image(image, shape=(256, 256)):
    image = image.resize(shape)
    image = img_to_array(image)
    image = np.dstack([image] * 3)

    return image


def get_image(image):
    image = image.astype(np.uint8)
    img = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img


class NumpyEncoder(json.JSONEncoder):
    '''
  Encoding numpy into json
  '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
