#! /usr/bin/env python

# Python imports
import os
import time
import math
import urllib.request as request
from multiprocessing.pool import ThreadPool
import argparse
from collections import OrderedDict

# Dependency Imports
from tqdm import tqdm
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials

download_batch_size = 256  # Limited to 256 by Custom Vision. See docs for CustomVisionTrainingClientOperationsMixin.get_tagged_images

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fetch data from CustomVision. Grab endpoint, training_key and project_id from https://www.customvision.ai/projects/<project_id>#/settings')
    parser.add_argument('--endpoint', type=str, 
        help='"Endpoint" from Custom Vision project settings, e.g. https://westus2.api.cognitive.microsoft.com/')
    parser.add_argument('--training_key', type=str, 
        help='"Key" from Custom Vision project settings, e.g. e46b53**************************')
    parser.add_argument('--project_id', type=str, 
        help='"Project Id" from Custom Vision project settings, e.g. 5539cc35-****-****-****-************')
    parser.add_argument('--output_directory', '-o', type=str, 
        help='e.g. "./downloads/"')

    args = parser.parse_args()
    
    endpoint = args.endpoint
    training_key = args.training_key
    project_id = args.project_id
    download_directory = args.output_directory
    
    # Prepare for Custom Vision download
    credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer = CustomVisionTrainingClient(endpoint, credentials)
    unique_tags = []

    image_count = trainer.get_tagged_image_count(project_id=project_id)
    progress_bar = tqdm(total=image_count)

    num_batches = math.ceil(image_count / download_batch_size)
    print(f"There will be {num_batches} batches downloaded ({image_count} images in total).")

    for batch_index in range(num_batches):
        image_batch = trainer.get_tagged_images(project_id=project_id, take=download_batch_size, skip=batch_index * download_batch_size)
        image_urls = []

        for index, image in enumerate(image_batch):
            base_filename = str((batch_index * download_batch_size) + index)

            lines = []
            for region in image.regions:
                # Register tag in unique_tags
                if region.tag_name not in unique_tags:
                    unique_tags.append(region.tag_name)
                # Get index of region tag in unique_tags
                tag_index = unique_tags.index(region.tag_name)
                # Construct YOLO line (format is "<object-class> <x-center> <y-center> <width> <height>", all numbers normalized between 0 and 1)
                line = f"{tag_index} {region.left + (region.width / 2)} {region.top  + (region.height / 2)} {region.width} {region.height}"
                lines.append(line)
    
            # Create data file
            data_filename = base_filename + ".txt"
            with open(os.path.join(download_directory, data_filename), "w+") as data_file:
                data_file.write("\n".join(lines))

            # Queue image URL for download
            image_download_url = image.original_image_uri
            image_destination_path = os.path.join(download_directory, base_filename + ".jpg")
            image_urls.append((image_download_url, image_destination_path))

        def download_url(data):
            request.urlretrieve(url=data[0], filename=data[1])
            progress_bar.update(1)
        
        # Download all image files in the batch simultaneously
        ThreadPool(download_batch_size).map(download_url, image_urls)

    progress_bar.close()
    
    # Save unique tags into class.names file
    with open(os.path.join(download_directory, "class.names"), "w+") as tags_file:
        tags_file.write("\n".join(unique_tags))

if __name__ == '__main__':
    print("Starting download...")
    time_start = time.time()
    main()
    time_end = time.time()
    time_elapsed_seconds = time_end - time_start
    print("Downloaded in %d minutes and %.2f seconds." % (time_elapsed_seconds // 60, time_elapsed_seconds % 60),)