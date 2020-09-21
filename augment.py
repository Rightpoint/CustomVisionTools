#! /usr/bin/env python

# Python imports
import time
import os
import argparse
from shutil import copyfile
from pathlib import Path

# Dependency imports
import numpy as np
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
import imageio

# Local imports
from helpers.augmentation import data_to_ia, ia_to_data, ImageData

class Augment:
    """
    Defines the augmentation process for an image.

    Parameters
    ----------
    name : string
        Anything you want. Arbitrary string that will be used to identify the operation and name the final image.
    operation : imgaug.augmenters.Augmenter
        Any imgaug list augmenter. The operations will be applied to the image.
        See https://imgaug.readthedocs.io/en/latest/source/overview/meta.html
    num_repetitions : int, optional
        The number of times `operation` will be applied, by default 1. Good for
        when you have a lot of randomness and need multiple passes.
    """
    def __init__(self, name, operation, num_repetitions=1):
        self.name = name
        self.operation = operation
        self.num_repetitions = num_repetitions

def make_default_ops():
    return [
        Augment("Blur", iaa.Sequential([
            iaa.GaussianBlur(sigma=(3.0, 5.0)) # Blur images with a sigma of 3.0 to 5.0
        ]), num_repetitions=3),
        Augment("AdditiveGaussianNoise", iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=0.05*255)
        ]), num_repetitions=1)
    ]

def make_rotation_ops(increments):
    ops = []
    for increment in increments:
        suffix = f"{increment}"
        if increment < 0:
            suffix = f"Back{-increment}"
        ops.append(
            Augment("Rotate" + suffix, iaa.Sequential([
                iaa.Grayscale(alpha=1.0),
                iaa.Rotate(increment)
            ]), num_repetitions=1)
        )
    return ops

def make_scale_ops():
    ops = []
    ops.append(
        Augment("Scale", iaa.Sequential([
            iaa.Grayscale(alpha=1.0),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                rotate=(-5, 5)
            )
        ]), num_repetitions=5),
        Augment("Original", iaa.Sequential([
            iaa.Grayscale(alpha=1.0),
        ]), num_repetitions=1),
    )
    return ops

def main():
    # Configure imgaug
    ia.seed(1)

    #########################################################
    # Provide a list of steps. Each one specifies a name 
    # suffix and an image transformation.
    #########################################################

    ops = [
        Augment("Scale", iaa.Sequential([
            iaa.SomeOf((0, 3), [
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
            ]),
        ]), num_repetitions=5),
    ] + make_default_ops() + make_rotation_ops([-15, -10, -5, 5, 10, 15]) + make_scale_ops()

    #########################################################
    # Parse arguments
    #########################################################

    parser = argparse.ArgumentParser(description='Applies a set of augmentations to every image in the input directory.')
    parser.add_argument('--input_directory', '-i', type=str, 
        help='e.g. "./downloads/"')
    parser.add_argument('--output_directory', '-o', type=str, 
        help='e.g. "./downloads/"')
    parser.add_argument('--multithreaded', '-m', action='store_true', 
        help='Process images in multiple threads')
    parser.add_argument('--preview_only', '-p', action='store_true', 
        help='Show previews instead of writing to disk')

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    user_requested_preview_only = args.preview_only
    multithreaded = args.multithreaded

    #########################################################
    # Process all data files in the input directory, either 
    # copying them over or queueing them to be augmented in 
    # the next step.
    #########################################################

    # Load YOLO region class names from file
    class_names = []
    with open(os.path.join(input_directory, "class.names")) as class_file:
        class_names = [line.rstrip() for line in class_file if line.rstrip() != ""]

    augment_files = []

    filenames = os.listdir(input_directory)
    filenames.sort()
    for filename in filenames:
        # Work only on YOLO txt files.
        if not filename.endswith(".txt"):
            continue

        if filename == "class.names":
            # Just copy the file over.
            if not user_requested_preview_only:
                copyfile(os.path.join(input_directory, filename), os.path.join(output_directory, filename))
            continue

        base_filename = os.path.splitext(filename)[0]
        data_path = os.path.join(input_directory, base_filename + ".txt")
        image_path = os.path.join(input_directory, base_filename + ".jpg")
        augment_files.append(ImageData(data_path, image_path))

        if not user_requested_preview_only:
            # Copy the original data file to the output directory.
            copyfile(
                os.path.join(input_directory, base_filename + ".txt"),
                os.path.join(output_directory, base_filename + ".txt"))
            # Copy the original image to the output directory.
            copyfile(
                os.path.join(input_directory, base_filename + ".jpg"),
                os.path.join(output_directory, base_filename + ".jpg"))

    #########################################################
    # From the list of data/image pairs to augment, create
    # batches for imgaug to process.
    #########################################################

    batch = []
    batches = []
    MAX_BATCH_SIZE = 10 if user_requested_preview_only else 100

    for i, item in enumerate(augment_files):
        item.populate_regions_from_yolo_data(class_names)

        # Load the image and bounding boxes into memory
        image, bbs = data_to_ia(item)
        batch.append((image, bbs, item))
        
        # If we're at the max batch size or the end of the file list,
        # finalize the batch and add it to the batch list. 
        if len(batch) == MAX_BATCH_SIZE or i == len(augment_files) - 1:
            images, bounding_boxes, data = list(zip(*batch))
            batches.append(UnnormalizedBatch(images=images, bounding_boxes=bounding_boxes, data=data))
            batch.clear()

    #########################################################
    # Apply each operation in ops to each image
    #########################################################

    for op in ops:
        for i in range (op.num_repetitions):
            # Produce augmentations
            for batches_aug in op.operation.augment_batches(batches, background=multithreaded and not user_requested_preview_only):
                if user_requested_preview_only:
                    # Preview output one batch at a time.
                    # Blocks execution until the window is closed.
                    # Closing a window will cause the next batch to appear.
                    # Close the Python instance in the dock to stop execution.
                    images_with_labels = [bb.draw_on_image(image) for image, bb in zip(batches_aug.images_aug, batches_aug.bounding_boxes_aug)]
                    grid_image = ia.draw_grid(images_with_labels, cols=None, rows=None)
                    title = f"{op.name}\nRep {i}\n"
                    # title += ", ".join([item.image_filename for item in batches_aug.data])  # Draw image filenames
                    grid_image = ia.draw_text(grid_image, 8, 8, title, color=(255, 0, 0), size=50)
                    ia.imshow(grid_image, backend='matplotlib')
                    continue

                for image, bbs, data in zip(batches_aug.images_aug, batches_aug.bounding_boxes_aug, batches_aug.data):
                    # Write image and matching data file to output folder

                    # Determine base name for image and matching data file
                    image_filename_no_extension, image_extension = os.path.splitext(Path(data.image_path).name)
                    base_filename = ""
                    if op.num_repetitions == 1:
                        base_filename = f"{image_filename_no_extension}_{op.name}"
                    else:
                        base_filename = f"{image_filename_no_extension}_{op.name}_rep{i}"

                    # Write image to output folder
                    output_image_path = os.path.join(output_directory, f"{base_filename}{image_extension}")
                    imageio.imwrite(output_image_path, image)

                    # Write modified imgaug bounding boxes as YOLO format in output folder
                    output_data_path = os.path.join(output_directory, f"{base_filename}.txt")
                    output_data = ia_to_data(output_data_path, output_image_path, image, bbs)

                    with open(output_data_path, "w+") as data_file:
                        lines = []
                        for region in output_data.regions:
                            # Construct YOLO line (format is "<object-class> <x-center> <y-center> <width> <height>", all numbers normalized between 0 and 1)
                            line = f"{class_names.index(region.tag_name)} {region.left + (region.width / 2)} {region.top + (region.height / 2)} {region.width} {region.height}"
                            lines.append(line)
                        data_file.write("\n".join(lines))

if __name__ == '__main__': # Need to do this or multithreading fails.
    print("Augmenting images...")
    time_start = time.time()
    main()
    time_end = time.time()
    print("Augmentation done in %.2fs" % (time_end - time_start,))
