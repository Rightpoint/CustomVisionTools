#! /usr/bin/env python

# Python imports
import time
import os
import math
from pathlib import Path
import argparse

# Dependency imports
from tqdm import tqdm
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, ImageFileCreateBatch, Region, CustomVisionErrorException, ImageCreateStatus
from msrest.authentication import ApiKeyCredentials

# Local imports
from helpers.augmentation import ImageData, ImageDataRegion

# Hard limit imposed by Custom Vision's API. 
# See docs for `create_images_from_files`
max_upload_batch_size = 64

# Name of the region tag used by the `--add_superfluous_regions` flag.
superfluous_tag_name = "coreml_bugfix"
num_images_tagged_with_superfluous_tag = 15  # Minimum required by Custom Vision

def image_data_region_to_custom_vision_region(region, map_tag_name_to_id):
    tag_id = map_tag_name_to_id.get(region.tag_name)
    if tag_id == None:
        raise Exception(f"No Custom Vision tag ID found for locally-defined region with tag name \"{region.tag_name}\".")
    return Region(
        tag_id=tag_id,
        left=region.left,
        top=region.top,
        width=region.width,
        height=region.height)

def image_data_to_custom_vision_create_entry(image_data, image_path, map_tag_name_to_id):
    image_file = open(image_path, mode="rb")
    custom_vision_regions = [image_data_region_to_custom_vision_region(region, map_tag_name_to_id) for region in image_data.regions]
    create_entry = ImageFileCreateEntry(name=Path(image_path).name, contents=image_file.read(), regions=custom_vision_regions)
    image_file.close()
    return create_entry

class LocalData:
    """
    Factory that produces what `upload_local_data()` needs.
    Given a base directory and a known annotation format, this
    class will load the data in a format that Custom Vision
    can understand.

    You can make more strategies for other annotation formats,
    such as VOC or COCO.
    """
    def __init__(self):
        self.image_data = []
        self.image_paths = []
        self.data_paths = []
        self.tags = []

    @staticmethod
    def load_yolo(base_directory):
        data = LocalData()

        # Load region class names from file
        class_names = []
        with open(os.path.join(base_directory, "class.names")) as class_file:
            class_names = [line.rstrip() for line in class_file if line.rstrip() != ""]
        data.tags = class_names

        # Populate from txt files in directory.
        data_filenames = [x for x in os.listdir(base_directory) if x.endswith(".txt")]
        for data_filename in data_filenames:
            data_path = os.path.join(base_directory, data_filename)
            data.data_paths.append(data_path)

            image_data = ImageData.from_yolo_data(data_path, class_names)
            data.image_data.append(image_data)

            data.image_paths.append(os.path.join(base_directory, data_filename.replace(".txt", ".jpg")))

        return data

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Upload data to Custom Vision. Grab endpoint, training_key and project_id values from https://www.customvision.ai/projects/<project_id>#/settings.')
    parser.add_argument('--endpoint', type=str, 
        help='"Endpoint" from Custom Vision project settings, e.g. https://westus2.api.cognitive.microsoft.com/')
    parser.add_argument('--training_key', type=str, 
        help='"Key" from Custom Vision project settings, e.g. e46b53**************************')
    parser.add_argument('--project_id', type=str, 
        help='"Project Id" from Custom Vision project settings, e.g. 5539cc35-****-****-****-************')
    parser.add_argument('--input_directory', '-i', type=str, 
        help='e.g. "./downloads/"')
    parser.add_argument('--add_superfluous_regions', 
        action='store_true', 
        help=f"Add a superflous tag to {num_images_tagged_with_superfluous_tag} image regions to fix a Custom Vision incompatibility with CoreML"
    )

    args = parser.parse_args()
    endpoint = args.endpoint
    training_key = args.training_key
    project_id = args.project_id
    input_image_directory = args.input_directory
    add_superfluous_regions = args.add_superfluous_regions
    
    # Prepare for Custom Vision upload
    credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer = CustomVisionTrainingClient(endpoint, credentials)

    def add_superfluous_region_to_regions(regions, map_tag_name_to_id):
        tag_id = map_tag_name_to_id.get(superfluous_tag_name)
        if tag_id == None:
            raise Exception(f"No Custom Vision tag ID found for superfluous region with tag name {superfluous_tag_name}.")
        regions.append(Region(
            tag_id=tag_id, 
            left=0, 
            top=0, 
            width=1, 
            height=1))

    def register_tag_names(tag_names):
        """
        Register local region tag names with Custom Vision and return 
        corresponding Custom Vision tag objects from the server.
        """
        for tag_name in tag_names:
            try:
                trainer.create_tag(project_id=project_id, name=tag_name)
            except CustomVisionErrorException as e:
                # If a tag with the given name already exists server-side, create_tag 
                # throws an exception. We ignore this behavior, since we only care 
                # about creating it if it isn't already registered.
                print(f"Exception: {e}")

        # Fetch all tags from the server
        package_tags = trainer.get_tags(project_id=project_id)
        return package_tags

    def process_batch(images):
        """
        Uploads a batch of images to Custom Vision.
        """
        batch = ImageFileCreateBatch(images=images)
        # Limited to 64 images and 20 tags per batch.
        upload_result = trainer.create_images_from_files(project_id=project_id, batch=batch)

        num_duplicates = 0
        error_results = []
        for result in upload_result.images:
            if result.status == ImageCreateStatus.ok_duplicate:  # Ignore errors indicating duplicate images
                num_duplicates += 1
            elif result.status != ImageCreateStatus.ok:
                error_results.append(result)

        if not upload_result.is_batch_successful and len(error_results) > 0:
            message = "Batch did not upload successfully!"
            for result in error_results:
                message += f"""
                url: {result.source_url}
                -- status: {result.status}
                """
            raise Exception(message)
        elif num_duplicates > 0:
            print(f"Batch uploaded successfully. Ignoring {num_duplicates} duplicates.")
        else:
            # print("Batch uploaded successfully.")
            pass

    def upload_local_data(local_data):
        """
        `local_data`: An array of LocalData objects.
        """

        # Register tags with Custom Vision
        tags_to_register = local_data.tags
        if add_superfluous_regions:
            tags_to_register.append(superfluous_tag_name)
        custom_vision_tags = register_tag_names(tags_to_register)
    
        # Construct a mapping from local tag name to Custom Vision tag ID.
        map_tag_name_to_id = {}
        for tag in custom_vision_tags:
            map_tag_name_to_id[tag.name] = tag.id

        total_batch_count = math.ceil(len(local_data.image_data) / max_upload_batch_size)
        current_batch_index = 0
        print(f"There will be {total_batch_count} batches uploaded ({len(local_data.image_data)} images in total).")

        custom_vision_data_batch = []

        progress_bar = tqdm(total=total_batch_count)

        # For each data file, convert the image file and region information 
        # to Custom Vision objects and upload them in batches.
        for i, (data, image_path) in enumerate(zip(local_data.image_data, local_data.image_paths)):
            custom_vision_data = image_data_to_custom_vision_create_entry(data, image_path, map_tag_name_to_id)  # Loads image and backing data into memory
            if add_superfluous_regions and i < num_images_tagged_with_superfluous_tag:
                """
                If the --add_superfluous_regions flag is set, then the first
                `num_images_tagged_with_superfluous_tag` images will get a 
                useless extra region added on. This fixes a bug with the 
                CoreML model that Custom Vision generates, where it seems
                to need at least two tags to function.
                """
                add_superfluous_region_to_regions(custom_vision_data.regions, map_tag_name_to_id)
            custom_vision_data_batch.append(custom_vision_data)
            
            # Upload the batch if we've hit the max batch size or the last data file.
            if len(custom_vision_data_batch) == max_upload_batch_size or i == len(local_data.image_data) - 1:
                # print(f"Uploading batch {current_batch_index + 1} of {total_batch_count}...")
                try:
                    process_batch(custom_vision_data_batch)
                except Exception as e:
                    print(f"Exception: {e}")
                custom_vision_data_batch.clear()
                current_batch_index += 1
                progress_bar.update(1)
        
        progress_bar.close()

    # Load local representations from disk
    local_data = LocalData.load_yolo(input_image_directory)

    # Schedule local data to upload to Custom Vision in batches
    upload_local_data(local_data)

if __name__ == '__main__':
    print("Starting upload...")
    time_start = time.time()
    main()
    time_end = time.time()
    time_elapsed_seconds = time_end - time_start
    print("Upload completed in %d minutes and %.2f seconds." % (time_elapsed_seconds // 60, time_elapsed_seconds % 60),)
