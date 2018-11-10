from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr

import os
import numpy as np
from six import raise_from
from PIL import Image


try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

rdd_classes = {
    'D00': 0,
    'D01': 1,
    'D10': 2,
    'D11': 3,
    'D20': 4,
    'D40': 5,
    'D43': 6,
    'D44': 7,
    'D30': 8
}


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class RDDGenerator(Generator):
    """ Generate data for Road damage detection
    """

    def __init__(
            self,
            data_dir,
            classes=rdd_classes,
            image_extension='.jpg',
            images_df=None,
            skip_truncated=False,
            skip_difficult=False,
            **kwargs
    ):
        """ Initialize a Pascal VOC data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """
        self.data_dir = data_dir
        self.classes = classes
        # self.image_names = [l.strip().split(None, 1)[0] for l in
        #                     open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]

        self.image_names = images_df.file.tolist()
        print(len(self.image_names))
        self.image_extension = image_extension
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(RDDGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        path = os.path.join(self.data_dir, "ImageSets",self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path = os.path.join(self.data_dir,"ImageSets",  self.image_names[image_index] + self.image_extension)
        return read_image_bgr(path)

    def __parse_annotation(self, element):
        """ Parse an annotation given an XML element.
        """
        # truncated = _findNode(element, 'truncated', parse=int)
        # difficult = _findNode(element, 'difficult', parse=int)
        truncated = 0
        difficult = 0

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box, label

    def __parse_annotations(self, xml_root):
        """ Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((len(xml_root.findall('object')),)),
                       'bboxes': np.empty((len(xml_root.findall('object')), 4))}
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box, label = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue

            annotations['bboxes'][i, :] = box
            annotations['labels'][i] = label

        return annotations

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
