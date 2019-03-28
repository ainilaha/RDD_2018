
This repository for Road [Damage Detection and Classification Challenges](https://bdc2018.mycityreport.net/overview/) with 
dataset collected by University of Tokyo and published in [https://doi.org/10.1111/mice.12387](https://doi.org/10.1111/mice.12387).
The implementation is based on [Keras RetinaNet](https://github.com/fizyr/keras-retinanet)

#### More detail can refer the paper
`@INPROCEEDINGS{8622025, 
author={L. Ale and N. Zhang and L. Li}, 
booktitle={2018 IEEE International Conference on Big Data (Big Data)}, 
title={Road Damage Detection Using RetinaNet}, 
year={2018}, 
volume={}, 
number={}, 
pages={5197-5200}, 
doi={10.1109/BigData.2018.8622025}, 
ISSN={}, 
month={Dec},}`

## Installation

1) Clone this repository.
2) Ensure numpy is installed using `pip install numpy --user`
3) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` and `pandas` are installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.

## Testing
Trained models mode can [Download here](https://www.dropbox.com/sh/dsj1wby8c5yhgsp/AAC1yWzaF-XZ0gS5ae2pYqKAa?dl=0)

An example of testing the network can be seen in [this Notebook](https://github.com/ainilaha/RDD_2018/blob/master/ResNet152RetinaNet.ipynb).
In general, inference of the network works as follows:
```python
boxes, scores, labels = model.predict_on_batch(inputs)
```

Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.

Loading models can be done in the following manner:
```python
from keras_retinanet.models import load_model
model = load_model('trained_models/model.h5', backbone_name='resnet152')
```


### Converting a training model to inference model
The training procedure of `keras-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
python keras_retinanet/bin/convert_model.py snapshots/training/model.h5 snapshots/inference/model.h5
```

Most scripts (like `retinanet-evaluate`) also support converting on the fly, using the `--convert-model` argument.


## Training
`keras-retinanet` can be trained using [this](https://github.com/ainilaha/RDD_2018/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.


The default backbone is `resnet50`. You can change this using the `--backbone=xxx` argument in the running script.
`xxx` can be one of the backbones in resnet models (`resnet50`, `resnet101`, `resnet152`), mobilenet models (`mobilenet128_1.0`, `mobilenet128_0.75`, `mobilenet160_1.0`, etc), densenet models or vgg models. The different options are defined by each model in their corresponding python scripts (`resnet.py`, `mobilenet.py`, etc).

Trained models can't be used directly for inference. To convert a trained model to an inference model, check [here](https://github.com/ainilaha/RDD_2018/blob/master/keras_retinanet/bin/convert_model.py).

### Usage
For training on Road Damage Dataset, the re-organized dataset can download 
[here](https://www.dropbox.com/s/zwd309g4u4derfi/road_damage_dataset.zip?dl=0).


Note: The original dataset was organized by locations. In order to ease to address the data we moved
all the the images into ImageSets sub folder ,and all the annotations are in Annotations sub folder.
In addition, we have created two CVS files to index data so that users can easily load data
preprocessing tools such as pandas.

If you have the original dataset please organize the dataset like above by moving images into one folder and 
annotations in another folder and then copy `data_index/trainset.cvs` and `testset.cvs` into folder of the dataset.
Therefore, the folder organize as follow:

road_damage_dataset\
&nbsp;&nbsp;|\
&nbsp;&nbsp;|--ImageSets\
&nbsp;&nbsp;|--Annotations\
&nbsp;&nbsp;|--trainset.cvs\
&nbsp;&nbsp;|--testset.cvs



run:
```shell
# Running directly from the repository:
python keras_retinanet/bin/train.py --backbone resnet152 rdd /path/to/road_damage_dataset

```

### Submitting Results

The default submission is backbone with ResNet152 with threshold confident 0.55.
You can change backbones and other parameters in `python submit_results.py`.


Run below command and produce csv file `submit_res152_55.csv` that can
submit to the competition platform.

