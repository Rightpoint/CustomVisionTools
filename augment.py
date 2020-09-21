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
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.batches import UnnormalizedBatch
import imageio

# Local imports
from helpers.augmentation import data_to_ia, ia_to_data, ImageData
from MyAugments import get_augmentation_operations

"""
This file is for the boilerplate around image augmentation. To modify the image
augmentation itself, modify MyAugments.py.
"""

def main():
    # Configure imgaug
    ia.seed(1)

    #########################################################
    # Parse arguments
    #########################################################

    parser = argparse.ArgumentParser(description='Applies a set of augmentations to every image in the input directory.')
    parser.add_argument('--input_directory', '-i', type=str, 
        help='e.g. "./downloads/"')
    parser.add_argument('--output_directory', '-o', type=str, 
        help='e.g. "./downloads/"')
    parser.add_argument('--single_threaded', '-s', action='store_true', 
        help='Process images in one thread instead of multithreading')
    parser.add_argument('--preview_only', '-p', action='store_true', 
        help='Show previews instead of writing to disk')
    parser.add_argument('--skip_originals', action='store_true', 
        help='Prevent original images from being copied into the destination folder')

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    user_requested_preview_only = args.preview_only
    single_threaded = args.single_threaded
    skip_originals = args.skip_originals

    #########################################################
    # Process all data files in the input directory, either 
    # copying them over or queueing them to be augmented in 
    # the next step.
    #########################################################

    # Load YOLO region class names from file
    class_names = []
    with open(os.path.join(input_directory, "class.names")) as class_file:
        class_names = [line.rstrip() for line in class_file if line.rstrip() != ""]

    # Copy the YOLO region class names file to the output directory
    if not user_requested_preview_only:
        copyfile(os.path.join(input_directory, "class.names"), os.path.join(output_directory, "class.names"))

    augment_files = []

    filenames = os.listdir(input_directory)
    filenames.sort()
    for filename in filenames:

        # Work only on YOLO .txt files.
        if not filename.endswith(".txt"):
            continue

        base_filename = os.path.splitext(filename)[0]
        data_path = os.path.join(input_directory, base_filename + ".txt")
        image_path = os.path.join(input_directory, base_filename + ".jpg")
        augment_files.append(ImageData(data_path, image_path))

        if not skip_originals and not user_requested_preview_only:
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
    # Apply each operation in MyAugments.py to each image
    #########################################################

    should_multithread = not single_threaded and not user_requested_preview_only

    for op in get_augmentation_operations():
        for i in range (op.num_repetitions):
            # Produce augmentations
            for batches_aug in op.operation.augment_batches(batches, background=should_multithread):
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
