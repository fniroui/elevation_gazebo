import cv2
import numpy as np
import matplotlib.pylab as plt
import math
from scipy import signal
from mpl_toolkits.mplot3d import axes3d
from PIL import Image
from random import randint
import fileinput
import argparse
import os
from scipy import signal


parser = argparse.ArgumentParser(description='map_gen')
parser.add_argument(
    '--num', '-n',
    default=1,
    metavar='N',
    help='Number of maps to generate (Default = 1).')
parser.add_argument(
    '--size', '-s',
    default=30,
    metavar='S',
    help='Size of the map in meters (Default = 10.0 m).')

def rotateImage(image, angle, height):  # expect degree, positive angle is counter clock wise
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags = cv2.INTER_LINEAR, borderValue = height)
    return result

def gaussianFilter(size, scale=0.15):
    '''
    Generate and return a 2D Gaussian function
    of dimensions (sizex,sizey)

    If sizey is not set, it defaults to sizex
    A scale can be defined to widen the function (default = 0.333)

    Parameters:
    size         :    size of the filter (pxls)
    filter_size  :    size of the filter (pxls)
    max          :    max. of generated value
    min          :    min. of generated value
    '''
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-scale * (x ** 2 / float(size) + y ** 2 / float(size)))
    return g / g.sum()


def randomDEM(img_size, filter_size, max, min):
    '''
    Generate a pseudo-DEM from random correlated noise.

    Parameters:
    img_size     :    size of the image (pxls)
    filter_size  :    size of the filter (pxls)
    max          :    max. of generated value
    min          :    min. of generated value
    '''
    dem1 = np.random.rand(img_size + 2 * filter_size, img_size + 2 * filter_size)
    demSmooth = signal.convolve(dem1, gaussianFilter(filter_size), mode='valid')
    demSmooth = (demSmooth - demSmooth.min()) / (demSmooth.max() - demSmooth.min())
    demSmooth = demSmooth * (max - min) + min
    return demSmooth


def plot(img, time = 5):
    plt.imshow(img, interpolation='nearest')
    plt.draw()
    plt.pause(time)


def padding(dem, size=1):
    '''
    Apply a border of size to a spatial dataset

    Return the padded data with the original centred in the array
    '''
    out = np.zeros([dem.shape[0] + 2 * size, dem.shape[1] + 2 * size])
    out[:, :] = np.max(dem) + 1
    out[size:-size, size:-size] = dem
    return out


def localMin(dem):
    '''
    We wish to return the location of the minimum grid value
    in a neighbourhood.

    We assume the array is 2D and defined (y,x)

    We return wx,wy which are the cell displacements in x and y directions.

    '''
    wy = np.zeros_like(dem).astype(int)
    wx = np.zeros_like(dem).astype(int)
    winx = np.ones([3, 3])
    for i in range(3):
        winx[:, i] = i - 1
    winy = winx.transpose()
    demp = padding(dem, size=1)
    for y in np.arange(1, demp.shape[0] - 1):
        for x in np.arange(1, demp.shape[1] - 1):
            win = demp[y - 1:y + 2, x - 1:x + 2]
            ww = np.where(win == np.min(win))
            whereX = winx[ww][0]
            whereY = winy[ww][0]
            wy[y - 1, x - 1] = whereY
            wx[y - 1, x - 1] = whereX
    return wx, wy


def borders(img, sizex, yS, yE, T, gap):
    '''
    Generates the borders seperating the 5 sections of the map.

    Parameters:
    img     :       the map
    sizex   :       width of the map
    yS      :       start of the actual map
    yE      :       end of the actual map
    T       :       border thickness
    gap     :       border openning for the robot to pass

    Work Needed to be done:
    Fix the naming. sizex and sizey are flipped in some cases.

    '''
    for x in range(1, 5):
        nGap = randint(1, 2)
        if nGap == 2:
            y1Start = randint(int(yS + T / 2 + gap / 2), int(yE - T / 2 - gap / 2))
            y2Start = randint(int(yS + T / 2 + gap / 2), int(yE - T / 2 - gap / 2))

            if y1Start > y2Start:
                y1Start, y2Start = y2Start, y1Start

            y1Start = int(y1Start - gap / 2)
            y2Start = int(y2Start - gap / 2)
            y1End = y1Start + gap
            y2End = y2Start + gap

            cv2.rectangle(img, (int(x * sizex / 5 - T / 2), yS), (int(x * sizex / 5 + T / 2), int(y1Start)), np.random.randint(0, 200), -1)

            if y1End < y2Start:
                cv2.rectangle(img, (int(x * sizex / 5 - T / 2), y1End), (int(x * sizex / 5 + T / 2), int(y2Start)), np.random.randint(0, 200),
                              -1)

            cv2.rectangle(img, (int(x * sizex / 5 - T / 2), int(y2End)), (int(x * sizex / 5 + T / 2), int(yE)), np.random.randint(0, 200), -1)
        elif nGap == 1:
            yStart = randint(int(yS + T / 2 + gap / 2), int(yE - T / 2 - gap / 2))
            yStart = int(yStart - gap / 2)
            yEnd = yStart + gap
            cv2.rectangle(img, (int(x * sizex / 5 - T / 2), yS), (int(x * sizex / 5 + T / 2), int(yStart)), np.random.randint(0, 200), -1)
            cv2.rectangle(img, (int(x * sizex / 5 - T / 2), int(yEnd)), (int(x * sizex / 5 + T / 2), yE), np.random.randint(0, 200), -1)

    return img

def createWorld(num, size, height, offset):
    '''
    Create worlds.

    Parameters:
    num       :     number of worlds to generate
    size      :     size of the map (m) 
    height    :     height of the map (m, highest point - lowest point)
    offset    :     vertical offset from the X-Y plane 
    '''

    # The map must be square with dimensions of 2^n + 1. 
    # The higher 'n' is, the more resolution the world will have (will be slower).
    n = 9
    dim = int(2**n+1)

    # Filter size
    filter_size = dim/2
    # Wall Thickness (0.2m)
    T = int(round(dim / size * 0.2))
    # 70% of the maximum height
    maxHeight = int(255 * 0.7)
    minHeight = 1

    script_dir = os.path.dirname(__file__)
    for i in range(1, num + 1):
        img = randomDEM(dim, filter_size, maxHeight, minHeight)
        cv2.rectangle(img, (0,0), (dim,dim), 255, T)
        # img  = platform(img, int(2*dim/size), 0, 200, 100)
        img = img.astype(np.uint8)
        plot(img, 1)

        img_path = os.path.join(script_dir, '..', 'worlds/models/maps/map_' + str(i) + '.png')
        # # print img_path
        world_in = os.path.join(script_dir, 'resources/test.world')
        world_out = os.path.join(script_dir, '..', 'worlds/map_' + str(i) + '.world')

        im = Image.fromarray(img)
        im.save(img_path)

        with open(world_in) as infile, open(world_out, 'w') as outfile:
            for line in infile:
                line = line.replace('t_name', 'map_' + str(i))
                line = line.replace('t_size_x', str(size))
                line = line.replace('t_size_y', str(size))
                line = line.replace('t_size_z', str(height))
                line = line.replace('t_pos_z', str(offset))
                outfile.write(line)
        print 'Map', i, 'created.'

if __name__ == '__main__':
    args = parser.parse_args()
    # Number of maps, size (m), height (m), vertical offset (m)
    createWorld(int(args.num), int(args.size), 3.0, -1.5)
