# CustomVisionTools

MacOS tools for uploading and downloading object detection training data sets and annotations from CustomVision.

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

Navigate to your CustomVision project's settings for the following values e.g. `https://www.customvision.ai/projects/<project_id>#/settings`

* endpoint
* training_key
* project_id

### Download

```
poetry run ./download.py --endpoint <endpoint> --training_key <training_key> --project_id <project_id> --output_directory ./downloads
```

### Upload

```
poetry run ./upload.py --endpoint <endpoint> --training_key <training_key> --project_id <project_id> --input_directory ./downloads
```

Additional flags:
* `--add_superfluous_regions`: Setting this flag adds an extra useless region to 15 images. 
    - If there's only one tag in a Custom Vision model and you export it to a CoreML model, it won't work when used. This fixes that bug.
    - Custom Vision requires a minimum of 15 images to be associated with a tag for it to get used.

### Augment

```
poetry run ./augment.py --input_directory ./downloads --output_directory ./augmented
```

This applies the augmentations specified in `augment.py` to every image in the input directory, and writes the original and the result to the output directory. You can then upload the whole directory using upload.py.

Additional flags:
* `--preview_only` or `-p`: Preview augmentations without writing to any files.
* `--multithreaded` or `-m`: Perform augmentations on multiple threads.

## Similar projects:
- https://kevinsaye.wordpress.com/2020/05/01/uploading-and-downloading-content-from-custom-computer-vision/
