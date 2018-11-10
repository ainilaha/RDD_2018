"""
Copyright 2018 vidosits (https://github.com/vidosits/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils import get_file

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class InceptionResNet(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a RDD_2018 model using the correct backbone.
        """
        return inception_resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        if self.backbone == 'inceptres':
            file_url="https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2" \
                     "_weights_tf_dim_ordering_tf_kernels_notop.h5"
            # checksum = '6d6bbae143d832006294945121d1f1fc'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file( '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
                         file_url, cache_subdir='models')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        backbone = self.backbone.split('_')[0]

        if backbone != "inceptres":
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones inception_resnet.'.format(backbone))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')


def inception_resnet_retinanet(num_classes, backbone='inception_resnet', inputs=None, modifier=None, **kwargs):
    """ Constructs a RDD_2018 model using a densenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('densenet121', 'densenet169', 'densenet201')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in RDD_2018 (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a DenseNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(600, 600, 3))

    # create the inception resnet backbone

    inception_resnet = InceptionResNetV2(input_tensor=inputs, include_top=False)
    if modifier:
        inception_resnet = modifier(inception_resnet)

    # create the full model
    # print(inception_resnet.summary())
    # layer_names = ["block35_10_mixed", "block17_20_mixed", "block8_10_mixed"]
    layer_names = ["mixed_6a","mixed_7a","conv_7b"]
    layer_outputs = [inception_resnet.get_layer(name).output for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)