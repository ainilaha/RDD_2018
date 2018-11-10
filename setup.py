import setuptools
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        'keras_retinanet.utils.compute_overlap',
        ['keras_retinanet/utils/compute_overlap.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setuptools.setup(
    name='keras-RDD_2018',
    version='0.5.0',
    description='Keras implementation of RetinaNet object detection.',
    url='https://github.com/fizyr/keras-RDD_2018',
    author='Hans Gaiser',
    author_email='h.gaiser@fizyr.com',
    maintainer='Hans Gaiser',
    maintainer_email='h.gaiser@fizyr.com',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet', 'six', 'scipy', 'cython', 'Pillow', 'opencv-python', 'progressbar2'],
    entry_points={
        'console_scripts': [
            'RDD_2018-train=keras_retinanet.bin.train:main',
            'RDD_2018-evaluate=keras_retinanet.bin.evaluate:main',
            'RDD_2018-debug=keras_retinanet.bin.debug:main',
            'RDD_2018-convert-model=keras_retinanet.bin.convert_model:main',
        ],
    },
    ext_modules=extensions,
    setup_requires=["cython>=0.28"]
)
