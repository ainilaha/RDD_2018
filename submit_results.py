import progressbar
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
import pandas as pd
import datetime

import numpy as np

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


keras.backend.tensorflow_backend.set_session(get_session())

base_path = "/work/BigDataDecomp/lale/road_damage_dataset"
test_csv_path = base_path + "/testset.csv"

# model_path = "trained_models/resnet101_rdd_20_best8103_infer.h5"
# model_path = "trained_models/resnet50_rdd_82_infer.h5"
# model_path = "trained_models/vgg19_rdd_10_best8279_infer.h5"
model_path = "trained_models/resnet152_rdd_19_best8140_infer.h5"
model = models.load_model(model_path, backbone_name='resnet152')

labels_to_names = {0: 'D00',
                   1: 'D01',
                   2: 'D10',
                   3: 'D11',
                   4: 'D20',
                   5: 'D40',
                   6: 'D43',
                   7: 'D44',
                   8: 'D30'}

test_df = pd.read_csv(test_csv_path)

start = datetime.datetime.now()

predict_df = pd.DataFrame(columns=["ImageId", "PredictionString"])
for img in progressbar.progressbar(list(test_df.file), prefix='Parsing annotations: '):
    img_path = base_path + '/ImageSets/{}.jpg'.format(img)
    image = read_image_bgr(img_path)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    boxes /= scale

    PredictionString = ''
    i = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if i < 6 and score > 0.55 and label < 8:
            i = i + 1
            color = label_color(label)
            b = box.astype(int)
            PredictionString += (str(label + 1) + ' ' + str(b).replace("]", "").replace("[", "").strip() + " ")
    #  get the top one if there is not boxes confident over 0.5
    if PredictionString == '':
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if (label < 8) and (label >= 0):
                b = box.astype(int)
                PredictionString += (str(label + 1) + ' ' + str(b).replace("]", "").replace("[", "").strip() + " ")
                break
    if PredictionString == '':
        PredictionString += '0 -1 -1 -1 -1 '

    end = datetime.datetime.now()
    elapsed = end - start
    print(elapsed.seconds)
    predict_df = predict_df.append({'ImageId': img + ".jpg",
                                    'PredictionString': PredictionString.strip().replace("  ", " ").replace("  ",
                                                                                                            " ").replace(
                                        "  ", " ")},
                                   ignore_index=True)

    predict_df = predict_df.sort_values("ImageId")
    predict_df.to_csv("submit_res152_55.csv", header=False, index=False)
