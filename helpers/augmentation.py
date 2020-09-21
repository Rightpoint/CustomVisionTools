import os
import json
import argparse
from shutil import copyfile

import numpy as np
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import imageio

class ImageDataRegion:
    def __init__(self, tag_name, left, top, width, height):
        """
        Represents a tagged region of an image.

        Parameters
        ----------
        tag_name : string
            Name for the region
        left : float
            Normalized X coordinate of the origin (top-left)
        top : float
            Normalized Y coordinate of the origin (top-left)
        width : float
            Normalized width of the region
        height : float
            Normalized height of the region
        """
        self.tag_name = tag_name
        self.left = left
        self.top = top
        self.width = width
        self.height = height

class ImageData:
    def __init__(self, data_path, image_path):
        """
        Describes an image and data file pair.

        Parameters
        ----------
        data_path : string
            Full path of the data file
        image_path : [type]
            Full path of the image file
        """
        self.data_path = data_path
        self.image_path = image_path
        self.regions = []
    
    def add_region(self, tag_name, left, top, width, height):
        self.regions.append(ImageDataRegion(tag_name, left, top, width, height))

    def populate_regions_from_yolo_data(self, class_names):
        """
        # Populate ImageDataRegions from data file using known YOLO format
        # (format is "<object-class> <x-center> <y-center> <width> <height>", all numbers normalized between 0 and 1)
        """
        # Construct regions
        data_lines = []
        with open(self.data_path) as data_file:
            data_lines = [line.rstrip() for line in data_file if line.rstrip() != ""]

        for line in data_lines:
            elements = line.split(" ")
            tag_index = int(elements[0])
            tag_name = class_names[tag_index]
            center_x, center_y, width, height = [float(x) for x in elements[1:]]
            self.add_region(tag_name, center_x - (width / 2), center_y - (height / 2), width, height)

def data_to_ia(data):
    """
    Convert from ImageData format to what imgaug expects.
    """
    # Load the image
    image = imageio.imread(data.image_path)
    image_height, image_width, _ = image.shape
    # print(image.shape)

    # Create ia bounding boxes from json
    regions = []
    for region in data.regions:
        bb = BoundingBox(
            x1=region.left * image_width,
            y1=region.top * image_height,
            x2=(region.left + region.width) * image_width,
            y2=(region.top + region.height) * image_height,
            label=region.tag_name)
        regions.append(bb)
    bbs = BoundingBoxesOnImage(regions, shape=image.shape)

    return (image, bbs)

def ia_to_data(data_path, image_path, image, bbs):
    """
    Convert from what imgaug produces to a json file in our custom format.
    """
    image_height, image_width, _ = image.shape

    data = ImageData(data_path, image_path)
    for bb in bbs:
        data.add_region(
            bb.label,
            float(bb.x1_int) / image_width,
            float(bb.y1_int) / image_height,
            float(bb.x2_int - bb.x1_int) / image_width,
            float(bb.y2_int - bb.y1_int) / image_height)

    return data