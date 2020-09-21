# CustomVisionTools

CustomVisionTools is a macOS command line toolset for working with [Custom Vision](https://www.customvision.ai/)'s object recognition training. These tools make it easy to upload or download images and bounding box data, as well as augmenting your images to make your trained models much more robust.

## Setup

You'll need Python 3.7 or higher and Homebrew installed.

Clone this repo. In the root of the repo, open the terminal and run the following:
```sh
brew install poetry # Dependency manager, see https://github.com/python-poetry/poetry#installation
poetry install # Install dependencies
```

Then you can start editing the code:
```sh
poetry shell
code . # Or whatever you want
```

## Usage

### Download

This downloads all the images and bounding boxes in the project into a folder. The data will be saved in YOLO Darknet format.

```
poetry run ./download.py --endpoint <endpoint> --training_key <training_key> --project_id <project_id> --output_directory ./downloads
```

Parameters:
* `--endpoint`: "Endpoint" from Custom Vision project settings.
* `--training_key`: "Key" from Custom Vision project settings.
* `--project_id`: "Project Id" from Custom Vision project settings.
* `--output-directory`: The directory that you want the data saved into.


### Upload

This uploads a folder to the specified project on Custom Vision.

```
poetry run ./upload.py --endpoint <endpoint> --training_key <training_key> --project_id <project_id> --input_directory ./downloads
```

Parameters:
* `--endpoint`: "Endpoint" from Custom Vision project settings.
* `--training_key`: "Key" from Custom Vision project settings.
* `--project_id`: "Project Id" from Custom Vision project settings.
* `--input-directory`: The directory that you want to upload from. The folder must be in YOLO Darknet format.

Additional flags:
* `--add_superfluous_regions`: Setting this flag adds an extra useless region to 15 images. 
    - If there's only one tag in a Custom Vision model and you export it to a CoreML model, it won't work when used. This fixes that bug.
    - Custom Vision requires a minimum of 15 images to be associated with a tag for it to get used.

### Augment

This applies the augmentations specified in `augment.py` to every image in the input directory using [imgaug](https://github.com/aleju/imgaug), and writes the original and the results to the output directory. You can then upload the whole directory using upload.py.

```
poetry run ./augment.py --input_directory ./downloads --output_directory ./augmented
```

Parameters:
* `--input-directory`: The directory that you want to augment. This must be a YOLO Darknet-formatted folder.
* `--output-directory`: The directory that you want the data saved into.

Additional flags:
* `--preview_only` or `-p`: Preview augmentations without writing to any files.
* `--multithreaded` or `-m`: Perform augmentations on multiple threads.

## Similar projects:
- https://kevinsaye.wordpress.com/2020/05/01/uploading-and-downloading-content-from-custom-computer-vision/
