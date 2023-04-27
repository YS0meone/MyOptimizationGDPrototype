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

import argparse
import multiprocessing
from pathlib import Path
import optimization
from global_optimization import global_optimize, auto_temp_schedule
# from scipy.ndimage import distance_transform_edt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from lineage_funcs import create_lineage, save_lineage
import jsonc
import sys
from itertools import count
from cell import Bacilli
import matplotlib.pyplot as plt
from copy import deepcopy
def parse_args():
    """Reads and parses the command-line arguments."""
    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument('-d', '--debug', metavar='DIRECTORY', type=Path, default=None,
                        help='path to the debug directory (enables debug mode)')
    parser.add_argument('-ff', '--frame_first', metavar='N', type=int, default=0,
                        help='starting image (default: %(default)s)')
    parser.add_argument('-lf', '--frame_last', metavar='N', type=int, default=-1,
                        help='final image (defaults to until last image)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='use distance-based objective function')
    parser.add_argument('-w', '--workers', type=int, default=-1,
                        help='number of parallel workers (defaults to number of processors)')
    parser.add_argument('-j', '--jobs', type=int, default=-1,
                        help='number of jobs per frame (defaults to --workers/-w)')
    parser.add_argument('--keep', type=int, default=1,
                        help='number of top solutions kept (must be equal or less than --jobs/-j)')
    parser.add_argument('--strategy', type=str, default='best-wins',
                        help='one of "best-wins", "worst-wins", "extreme-wins"')
    parser.add_argument('--cluster', type=str, default='',
                        help='dask cluster address (defaults to local cluster)')
    parser.add_argument('--no_parallel', action='store_true', default=False, help='disable parallelism')
    parser.add_argument('--global_optimization', action='store_true', default=False, help='global optimization')
    parser.add_argument('--binary', action='store_true', default=True,
                        help="input image is binary")
    parser.add_argument('--graySynthetic', action='store_true', default=False,
                        help='enables the use of the grayscale synthetic image for use with non-thresholded images')
    parser.add_argument('--phaseContrast', action='store_true', default=False,
                        help='enables the use of the grayscale synthetic image for phase contract images')
    parser.add_argument('-ta', '--auto_temp', metavar='TEMP', type=int, default=1,
                        help='auto temperature scheduling for the simulated annealing')
    parser.add_argument('-ts', '--start_temp', type=float, help='starting temperature for the simulated annealing')
    parser.add_argument('-te', '--end_temp', type=float, help='ending temperature for the simulated annealing')
    parser.add_argument('-am', '--auto_meth', type=str, default='none', choices=('none', 'frame', 'factor', 'const', 'cost'),
                        help='method for auto-temperature scheduling')
    parser.add_argument('-r', "--residual", metavar="FILE", type=Path, required=False,
                        help="path to the residual image output directory")
    parser.add_argument('--lineage_file', metavar='FILE', type=Path, required=False,
                        help='path to previous lineage file')
    parser.add_argument('--continue_from', metavar='N', type=int, default=0,
                        help="load already found orientation of cells and start from the continue_from frame")
    parser.add_argument('--seed', metavar='N', type=int, default=None, help='seed for random number generation')
    parser.add_argument('--batches', metavar='N', type=int, default=1, help='number of batches to split each frame into for multithreading')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', metavar='PATTERN', type=str, required=True,
                          help='input filename pattern (e.g. "image%%03d.png")')
    required.add_argument('-o', '--output', metavar='DIRECTORY', type=Path, required=True,
                          help='path to the output directory')
    required.add_argument('-c', '--config', metavar='FILE', type=Path, required=True,
                          help='path to the configuration file')
    required.add_argument('-x', '--initial', metavar='FILE', type=Path, required=True,
                          help='path to the initial cell configuration')
    required.add_argument('-b', "--bestfit", metavar="FILE", type=Path, required=True,
                          help="path to the best fit synthetic image output directory")

    parsed = parser.parse_args()

    if parsed.workers == -1:
        parsed.workers = multiprocessing.cpu_count()

    if parsed.jobs == -1:
        if parsed.cluster:
            raise ValueError('-j/--jobs is required for non-local clusters')
        else:
            parsed.jobs = parsed.workers

    return parsed


def load_config(config_file):
    """Loads the configuration file."""
    with open(config_file) as fp:
        config = jsonc.load(fp)

    if not isinstance(config, dict):
        raise ValueError('Invalid config: must be a dictionary')
    elif 'global.cellType' not in config:
        raise ValueError('Invalid config: missing "global.cellType"')
    elif 'global.pixelsPerMicron' not in config:
        raise ValueError('Invalid config: missing "global.pixelsPerMicron"')
    elif 'global.framesPerSecond' not in config:
        raise ValueError('Invalid config: missing "global.framesPerSecond"')

    if config['global.cellType'].lower() == 'bacilli':
        celltype = Bacilli
    else:
        raise ValueError('Invalid config: unsupported cell type')

    celltype.checkconfig(config)

    return config


def get_inputfiles(args):
    """Gets the list of images that are to be analyzed."""
    inputfiles = []

    if args.frame_first > args.frame_last and args.frame_last >= 0:
        raise ValueError('Invalid interval: frame_first must be less than frame_last')
    elif args.frame_first < 0:
        raise ValueError('Invalid interval: frame_first must be greater or equal to 0')

    for i in count(args.frame_first):
        # check to see if the file exists
        file = Path(args.input % i)
        if file.exists() and file.is_file():
            inputfiles.append(file)
            if i == args.frame_last:
                break
        elif args.frame_last < 0 and args.frame_first != i:
            break
        else:
            raise ValueError(f'Input file not found "{file}"')
    return inputfiles

def objective(realimage, synthimage):
    """Full objective function between two images."""
    return np.sum(np.square((realimage - synthimage)))

def get_loss(lineage, realimage):
    synthimage, _ = optimization.generate_synthetic_image(lineage.frames[0].nodes, shape, lineage.frames[0].simulation_config)
    loss = objective(realimage, synthimage)
    # print("Before gradient descent the objective function value is:", loss)
    return loss

def get_gradient(lineage, realimage):
    lineage_copy = deepcopy(lineage)
    gradient = [0]*5

    delta = 0.1
    
    f0 = get_loss(lineage_copy, realimage)
    lineage_copy.frames[0].nodes[0].cell.x += delta
    f1 = get_loss(lineage_copy, realimage)
    gradient[0] = (f1 - f0) / delta
    lineage_copy.frames[0].nodes[0].cell.x -= delta

    f0 = get_loss(lineage_copy, realimage)
    lineage_copy.frames[0].nodes[0].cell.y += delta
    f1 = get_loss(lineage_copy, realimage)
    gradient[1] = (f1 - f0) / delta
    lineage_copy.frames[0].nodes[0].cell.y -= delta

    f0 = get_loss(lineage_copy, realimage)
    lineage_copy.frames[0].nodes[0].cell.rotation += delta
    f1 = get_loss(lineage_copy, realimage)
    gradient[2] = (f1 - f0) / delta
    lineage_copy.frames[0].nodes[0].cell.rotation -= delta

    f0 = get_loss(lineage_copy, realimage)
    lineage_copy.frames[0].nodes[0].cell.length += delta
    f1 = get_loss(lineage_copy, realimage)
    gradient[3] = (f1 - f0) / delta
    lineage_copy.frames[0].nodes[0].cell.length -= delta

    f0 = get_loss(lineage_copy, realimage)
    lineage_copy.frames[0].nodes[0].cell.width += delta
    f1 = get_loss(lineage_copy, realimage)
    gradient[4] = (f1 - f0) / delta
    lineage_copy.frames[0].nodes[0].cell.width -= delta

    return gradient

# give you the lineage file of the current cells and the real image that you try to achieve
# gradient descent can traverse through all of the current cells and perturb them towards the outcome
def gradient_descent(lineage, realimage):
    loss = get_loss(lineage, realimage)
    # get the gradient vector for x, y, rotation, length, width
    gradient = get_gradient(lineage, realimage)
    print(gradient)
    

if __name__ == "__main__":
    
    # plt.imshow(images[0], cmap="gray")
    # plt.show()
    # how to use generate synthetic image function?
    # how does the original algorithm calculate the objective function
    # First generate the initial synthetic image
    args = parse_args()
    config = load_config(args.config)
    simulation_config = config["simulation"]
    if args.graySynthetic:
        simulation_config["image.type"] = "graySynthetic"
    elif args.phaseContrast:
        simulation_config["image.type"] = "phaseContrastImage"
    elif args.binary:
        simulation_config["image.type"] = "binary"
    
    config = load_config(args.config)
    imagefiles = get_inputfiles(args)
    realimages = [optimization.load_image(imagefile) for imagefile in imagefiles]
    # plt.imshow(realimages[0], cmap="gray")
    # plt.show()
    shape = realimages[0].shape
    lineage = create_lineage(imagefiles, realimages, config, args)
    print()
    gradient_descent(lineage, realimages[0])

    
    # plt.imshow(synthimage, cmap="gray")
    # plt.figure()
    # plt.imshow(realimages[0], cmap="gray")
    # plt.show()
    

#45.471131483105324