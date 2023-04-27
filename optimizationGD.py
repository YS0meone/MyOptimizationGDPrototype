# What can CellUniverse do?
# It can take the initial.csv and generate a synthetic image according to the parameters inside the initial.csv
# Can it take a real image and generate a synthetic image? 
# It should be able to take a real image and generate a synthetic image but CellUniverse won't know how many
# cell nodes there are in the synthetic image and what shape and orientation they are
# TODO: Get the generate synthetic image from cell universe and generate synthetic images of
# frame0 and frame1
# TODO: Be able to generate synthetic image from abstract data (given cellnode and background color)
# TODO: After getting the cellnode info of previous frame and the synthetic image of the next frame we can create the 
# objective function by subtracting two

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

def load_image(imagefile):
    """Open the image file and convert to a floating-point grayscale array."""
    with open(imagefile, "rb") as fp:
        realimage = np.array(Image.open(fp))
    if realimage.dtype == np.uint8:
        realimage = realimage.astype(float) / 255
    if len(realimage.shape) == 3:
        realimage = np.mean(realimage, axis=-1)
    return realimage

def generate_synthetic_image(cellnodes, shape, simulation_config):
    image_type = simulation_config["image.type"]
    cellmap = np.zeros(shape, dtype=int)
    if image_type == "graySynthetic" or image_type == "phaseContrast":
        background_color = simulation_config["background.color"]
        synthimage = np.full(shape, background_color)
        for node in cellnodes:
            node.cell.draw(synthimage, cellmap, is_cell, simulation_config)
        return synthimage, cellmap
    else:
        synthimage = np.zeros(shape)
        for node in cellnodes:
            node.cell.draw(synthimage, cellmap, is_cell, simulation_config)
        return synthimage, cellmap


if __name__ == "__main__":
    IMAGEFILES = ["frame000.png", "frame001.png"]
    images = [load_image(image) for image in IMAGEFILES]
    # plt.imshow(simages[0], cmap="gray")
    # plt.show()