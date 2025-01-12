import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt

from PIL import Image
from imgaug import augmenters as iaa
from helper_functions import read_leads


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_directory", type=str, required=True)
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-o", "--output_directory", type=str, required=True)
    parser.add_argument("-n", "--noise", type=int, default=25)
    parser.add_argument("-t", "--temperature", type=int, default=6500)
    return parser


# Main function for running augmentations
def get_augment(
    input_file,
    output_directory,
    rotate=25,
    noise=25,
    crop=0.01,
    temperature=6500,
    bbox=False,
    store_text_bounding_box=False,
    json_dict=None,
):
    filename = input_file
    image = Image.open(filename)

    image = np.array(image)

    lead_bbs = []
    leadNames_bbs = []

    (
        lead_bbs,
        leadNames_bbs,
        lead_bbs_labels,
        startTime_bbs,
        endTime_bbs,
        plotted_pixels,
    ) = read_leads(json_dict["leads"])

    images = [image[:, :, :3]]
    h, w, _ = image.shape
    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)
    # Augment in a sequential manner. Create an augmentation object
    seq = iaa.Sequential(
        [
            iaa.Affine(rotate=rot),
            iaa.AdditiveGaussianNoise(scale=(noise, noise)),
            iaa.Crop(percent=crop_sample),
            iaa.ChangeColorTemperature(temperature),
        ]
    )

    images_aug = seq(images=images)

    head, tail = os.path.split(filename)

    f = os.path.join(output_directory, tail)
    plt.imsave(fname=f, arr=images_aug[0])

    return f
