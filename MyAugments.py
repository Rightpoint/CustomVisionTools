# Dependency Imports
import imgaug as ia
from imgaug import augmenters as iaa

# Local imports
from helpers.augmentation import Augment

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
    return [
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
    ]

#########################################################
# Provide a list of steps. Each one specifies a name 
# suffix and an image transformation.
#########################################################
def get_augmentation_operations():
    """
    Returns a list of `Augment`s. For each `Augment` in the list, an image will be generated and modified by that `Augment`.

    Returns
    -------
    [Augment]
        A list of `Augment`s.
    """
    return [
        Augment("SimulateVariedCameraConditions", iaa.Sequential([
            iaa.SomeOf((1, 3), [ # Pick between 1 and 3 of the following to apply to the image
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # Increase sharpness by a random amount
                iaa.Add((-10, 10), per_channel=0.5), # Randomly adjust brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # Randomly adjust hue and saturation (by -20 to 20 of original value)
            ]),
        ]), num_repetitions=5),
    ] + make_default_ops() + make_rotation_ops([-15, -10, -5, 5, 10, 15]) + make_scale_ops()