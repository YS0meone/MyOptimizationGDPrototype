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
import time
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
import os
from scipy.ndimage import distance_transform_edt
from scipy.optimize import leastsq

# DUMMY_CONFIG = {'image.type': 'graySynthetic', 'background.color': 0.39186227304616866, 'cell.color': 0.20192893848498444, \
#     'light.diffraction.sigma': 16.49986536658998, 'light.diffraction.strength': 0.4625857370407187, 'light.diffraction.truncate': 1, \
#         'cell.opacity': 0.28680135149498254, 'padding': 0}
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

def get_loss(cellNodes, realimage, config):
    synthimage, _ = optimization.generate_synthetic_image(cellNodes, shape, config["simulation"])
    loss = objective(realimage, synthimage)
    # print("Before gradient descent the objective function value is:", loss)
    return loss

def show_synth_real(cellNodes, realimage, i, config):
    synthimage, _ = optimization.generate_synthetic_image(cellNodes, shape, config["simulation"])
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(synthimage, cmap="gray")
    ax[1].imshow(realimage, cmap="gray")
    # print(os.getcwd())
    plt.savefig("E:/CS/ForkedRepo/MyOptimizationGDPrototype/video0/real_synth/step{}afterGD.png".format(i))
    plt.close(fig)

def show_bestfit(cellNodes, realimage,i, config):
    synthimage, _ = optimization.generate_synthetic_image(cellNodes, shape, config["simulation"])
    bestfit_frame = Image.fromarray(np.uint8(255 * synthimage), "L")
    bestfit_frame.save("E:/CS/ForkedRepo/MyOptimizationGDPrototype/video0/bestfit/frame{:03d}.png".format(i))


def drawOutlines(cellNodes, realimage, i):
    shape = realimage.shape
    output_frame = np.empty((shape[0], shape[1], 3))
    output_frame[..., 0] = realimage
    output_frame[..., 1] = output_frame[..., 0]
    output_frame[..., 2] = output_frame[..., 0]
    for node in cellNodes:
        node.cell.drawoutline(output_frame, (1, 0, 0))
    output_frame = Image.fromarray(np.uint8(255 * output_frame))
    output_frame.save("E:/CS/ForkedRepo/MyOptimizationGDPrototype/video0/outlines/frame{}.png".format(i))

def get_gradient(cellNodes, realimage, config):
    nodes_copy = deepcopy(cellNodes)
    cell_cnt = len(nodes_copy)
    gradient = np.array([[0]*5 for i in range(cell_cnt)], dtype=np.float64)

    delta = 1
    f0 = get_loss(nodes_copy, realimage, config)
    for i in range(cell_cnt):
        
        nodes_copy[i].cell.x += delta
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][0] = (f1 - f0) / delta
        nodes_copy[i].cell.x = cellNodes[i].cell.x

        
        nodes_copy[i].cell.y += delta
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][1] = (f1 - f0) / delta
        nodes_copy[i].cell.y = cellNodes[i].cell.y 

        nodes_copy[i].cell.rotation +=  0.3
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][2] = (f1 - f0) / 0.3
        nodes_copy[i].cell.rotation = cellNodes[i].cell.rotation

        x = np.linspace(nodes_copy[i].cell.length - 2, nodes_copy[i].cell.length + 2, 50)
        print_loss_func(x, nodes_copy, realimage, i, config)
        d = nodes_copy[i].cell.length * 0.2
        nodes_copy[i].cell.length *= 1.2
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][3] = (f1 - f0) / d
        nodes_copy[i].cell.length = cellNodes[i].cell.length

        d = nodes_copy[i].cell.width * 0.2
        nodes_copy[i].cell.width *= 1.2
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][4] = (f1 - f0) / d
        nodes_copy[i].cell.width = cellNodes[i].cell.width

    return gradient

def modify_cells(cellNodes, step):
    nodes_copy = deepcopy(cellNodes)
    cell_cnt = len(nodes_copy)
    for i in range(cell_cnt):
        nodes_copy[i].cell.x += step[i][0]
        nodes_copy[i].cell.y += step[i][1]
        nodes_copy[i].cell.rotation += step[i][2]
        nodes_copy[i].cell.length += step[i][3]
        nodes_copy[i].cell.width += step[i][4]
    return nodes_copy

def modify_cell(cellNodes, step, i):
    nodes_copy = deepcopy(cellNodes)
    nodes_copy[i].cell.x += step[i][0]
    nodes_copy[i].cell.y += step[i][1]
    nodes_copy[i].cell.rotation += step[i][2]
    nodes_copy[i].cell.length += step[i][3]
    nodes_copy[i].cell.width += step[i][4]
    return nodes_copy

# this returns the derivative of loss function at alpha = a
def get_derivative(cellNodes, realimage, a, direction, config):
    delta = 0.03
    # direction here is numpy array and contains the negative gradient direction of all cells
    f1 = get_loss(modify_cells(cellNodes, direction * (a + delta)), realimage, config)
    f0 = get_loss(modify_cells(cellNodes, direction * a), realimage, config)
    return (f1 - f0)/delta

# this function does the line search for the optimized alpha
def secant_method(cellNodes, realimage, direction, config):
    a0 = 0.0
    a1 = 0.1
    while abs(a1 - a0) > 0.03:
        df0 = get_derivative(cellNodes, realimage, a0, direction, config)
        df1 = get_derivative(cellNodes, realimage, a1, direction, config)
        ddf = (df1 - df0) / (a1 - a0)
        a = a1 - df1/ddf
        a0 = a1
        a1 = a
    return a1


def backtrackLS(cellNodes, realimage, direction, config):
    cell_cnt = len(cellNodes)
    t = [2] * cell_cnt
    f0 = get_loss(cellNodes, realimage, config)
    beta = 0.8
    for i in range(cell_cnt):
        LHS = get_loss(modify_cell(cellNodes, t[i] * direction, i), realimage, config)
        RHS =  f0 - t[i]/2*(np.linalg.norm(direction[i]))**2
        # print(LHS, RHS)
        while LHS > RHS and t[i] > 0.05:
            t[i] = beta * t[i]
            LHS = get_loss(modify_cell(cellNodes, t[i] * direction, i), realimage, config)
            RHS = f0 - t[i]/2*(np.linalg.norm(direction[i]))**2
        # print(LHS, RHS)
        direction[i] = direction[i] * t[i]
    # print(t)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def find_optimal_simulation_conf(simulation_config, realimage1, cellnodes):
    shape = realimage1.shape

    def cost(values, target, simulation_config):
        for i in range(len(target)):
            simulation_config[target[i]] = values[i]
        synthimage, cellmap = optimization.generate_synthetic_image(cellnodes, shape, simulation_config)
        return (realimage1 - synthimage).flatten()

    initial_values = []
    variables = []

    variables.append("background.color")
    initial_values.append(simulation_config["background.color"])

    variables.append("cell.color")
    initial_values.append(simulation_config["cell.color"])

    variables.append("light.diffraction.sigma")
    initial_values.append(simulation_config["light.diffraction.sigma"])

    variables.append("light.diffraction.strength")
    initial_values.append(simulation_config["light.diffraction.strength"])

    auto_opacity = True
    variables.append("cell.opacity")
    initial_values.append(simulation_config["cell.opacity"])
    if len(variables) != 0:
        residues = lambda x: cost(x, variables, simulation_config)
        optimal_values, _ = leastsq(residues, initial_values)

        for i, param in enumerate(variables):
            simulation_config[param] = optimal_values[i]
        simulation_config["cell.opacity"] = max(0, simulation_config["cell.opacity"])
        simulation_config["light.diffraction.sigma"] = max(0, simulation_config["light.diffraction.sigma"])

        if auto_opacity:
            for node in cellnodes:
                node.cell.opacity = simulation_config["cell.opacity"]

    print(simulation_config)
    return simulation_config

# assume we give the copy of cellNodes
def print_loss_func(x, cellNodes, realimage, i, config):
    y = []
    nodes_copy = deepcopy(cellNodes)
    for val in x:
        nodes_copy[i].length = val
        y.append(get_loss(nodes_copy, realimage, config))
    plt.plot(x, y)
    plt.title(f"Cell {i}'s loss function of length")
    plt.show()


# give you the lineage file of the current cells and the real image that you try to achieve
# gradient descent can traverse through all of the current cells and perturb them towards the outcome
def gradient_descent(cellNodes, realimages, config):
    np.set_printoptions(precision=6, floatmode='fixed', suppress=True)
    # start_time = time.time()
    LOG_PATH = "E:/CS/ForkedRepo/MyOptimizationGDPrototype/video0/log.txt"
    log_file = open(LOG_PATH, 'w')
    for i in range(len(realimages)):
        # print()
        # print(config["simulation"])
        # print()
        # find_optimal_simulation_conf(config["simulation"], realimages[i], cellNodes)
        # print()
        loss0 = get_loss(cellNodes, realimages[i], config)
        print("Step {}\nLoss before GD is:{:.5f}".format(i, loss0))
        # log_file.write("Step {}\nLoss before GD is:{:.5f}\n".format(i, loss0))
        epoch = 60
        print("Step {}".format(i), file=log_file)
        while epoch > 0:
            
            gradient = get_gradient(cellNodes, realimages[i], config)
            
            
            direction = np.array([-1 * normalize(v) for v in gradient])
            #direction = -1 * gradient
            # np.savetxt("log.txt", direction, fmt='%.6e')

            backtrackLS(cellNodes, realimages[i], direction, config)

            print(direction, file=log_file)
            print("\n",file=log_file)
            

            # alpha = secant_method(cellNodes, realimages[i], direction)
            # alpha = 0.05
            # print(direction)
            # print(alpha)
            # step = alpha * direction
            # print(step)
            # get the gradient vector for x, y, rotation, length, width
            cellNodes = modify_cells(cellNodes, direction)
            # show_synth_real(cellNodes, realimages[i])
            loss1 = get_loss(cellNodes, realimages[i], config)
            # if loss1 == loss0:
            #     continue
            if abs(loss1 - loss0) < 0.001:
                break
            loss0 = loss1
            # print(loss1)
            epoch -= 1
        drawOutlines(cellNodes, realimages[i], i)
        show_bestfit(cellNodes, realimages[i], i, config)
        print("\n",file=log_file)
        print("Loss after GD is: {:.5f}".format(loss1))
        # log_file.write("Loss after GD is: {}\n\n".format(loss1))
    log_file.close()
    # print(f"------running time {time.time() - start_time} seconds------")
    
    

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
    
    imagefiles = get_inputfiles(args)
    realimages = [optimization.load_image(imagefile) for imagefile in imagefiles]
    # plt.imshow(realimages[0], cmap="gray")
    # plt.show()
    shape = realimages[0].shape
    # print(shape)
    lineage = create_lineage(imagefiles, realimages, config, args)
    config["simulation"] = lineage.frames[0].simulation_config
    cellNodes = lineage.frames[0].nodes
    # print()
    # # print(lineage.frames[0].simulation_config)

    # # synthimage, _ = optimization.generate_synthetic_image(cellNodes, shape, DUMMY_CONFIG)
    # # plt.imshow(synthimage, cmap="gray")
    # # plt.show()
    gradient_descent(cellNodes, realimages, config)

    # synthimage, _ = optimization.generate_synthetic_image(lineage.frames[0].nodes, shape, lineage.frames[0].simulation_config)
    # plt.imshow(synthimage, cmap="gray")
    # plt.figure()
    # plt.imshow(realimages[0], cmap="gray")
    # plt.show()
    

#45.471131483105324