import os
import json
import argparse
from shutil import copyfile

import numpy as np
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import imageio

class Augment:
    """
    Defines the augmentation process for an image.

    Parameters
    ----------
    name : string
        Anything you want. Arbitrary string that will be used to identify the operation and name the final image.
    operation : imgaug.augmenters.Augmenter
        Any imgaug list augmenter. The operations will be applied to the image.
        See https://imgaug.readthedocs.io/en/latest/source/overview/meta.html for more.
    num_repetitions : int, optional
        The number of times `operation` will be applied, by default 1. Good for
        when you have a lot of randomness and need multiple passes.
    """
    def __init__(self, name, operation, num_repetitions=1):
        self.name = name
        self.operation = operation
        self.num_repetitions = num_repetitions

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

    @staticmethod
    def from_imgaug(bb, image_width, image_height):
        # Convert from imgaug bounding box to ImageDataRegion
        return ImageDataRegion(
            bb.label,
            float(bb.x1_int) / image_width,
            float(bb.y1_int) / image_height,
            float(bb.x2_int - bb.x1_int) / image_width,
            float(bb.y2_int - bb.y1_int) / image_height)
    
    def to_imgaug(self, image_width, image_height):
        return BoundingBox(
            x1=self.left * image_width,
            y1=self.top * image_height,
            x2=(self.left + self.width) * image_width,
            y2=(self.top + self.height) * image_height,
            label=self.tag_name)

class ImageData:
    def __init__(self):
        """
        Holds framework-independent data about an image's tagged regions.
        """
        self.regions = []
    
    def add_region(self, tag_name, left, top, width, height):
        self.regions.append(ImageDataRegion(tag_name, left, top, width, height))

    @staticmethod
    def from_yolo_data(data_path, class_names):
        """
        Populate ImageDataRegions from data file using known YOLO format.
        
        Format is `<object-class> <x-center> <y-center> <width> <height>`, 
        with all numbers normalized between 0 and 1.

        Parameters
        ----------
        data_path : string
            Path to YOLO data file.
        class_names : [type]
            List of region class names.

        Returns
        -------
        ImageData
            ImageData populated from YOLO data.
        """
        data = ImageData()

        # Construct regions
        data_lines = []
        with open(data_path) as data_file:
            data_lines = [line.rstrip() for line in data_file if line.rstrip() != ""]

        for line in data_lines:
            elements = line.split(" ")
            tag_index = int(elements[0])
            tag_name = class_names[tag_index]
            center_x, center_y, width, height = [float(x) for x in elements[1:]]
            data.add_region(tag_name, center_x - (width / 2), center_y - (height / 2), width, height)

        return data

    def to_imgaug(self, image_shape):
        """
        Get imgaug.BoundingBoxesOnImage from ImageDataRegions.
        """
        image_height, image_width, _ = image_shape

        # Create ia bounding boxes from json
        regions = []
        for region in self.regions:
            regions.append(region.to_imgaug(image_width, image_height))
        bbs = BoundingBoxesOnImage(regions, shape=image_shape)

        return bbs

    @staticmethod
    def from_imagaug(image_width, image_height, imgaug_bounding_boxes):
        """
        Create from imgaug data.

        Parameters
        ----------
        image_width : int
            Width of image in pixels.
        image_height: int
            Height of image in pixels.
        imgaug_bounding_boxes : [imgaug.BoundingBox]
            List of bounding boxes in imgaug format.

        Returns
        -------
        ImageData
            ImageData populated from imgaug data.
        """

        data = ImageData()
        for bb in imgaug_bounding_boxes:
            # Convert from imgaug bounding boxes to ImageDataRegion
            data.regions.append(ImageDataRegion.from_imgaug(bb, image_width, image_height))

        return data

    def write_yolo(self, data_path, class_names):
        """
        Write image data as YOLO formatted data file.

        Parameters
        ----------
        data_path : string
            Path for new data file.
        class_names : [string]
            Region class name list.
        """
        with open(data_path, "w+") as data_file:
            lines = []
            for region in self.regions:
                # Construct YOLO line (format is "<object-class> <x-center> <y-center> <width> <height>", all numbers normalized between 0 and 1)
                line = f"{class_names.index(region.tag_name)} {region.left + (region.width / 2)} {region.top + (region.height / 2)} {region.width} {region.height}"
                lines.append(line)
            data_file.write("\n".join(lines))