# Modify https://github.com/BMIRDS/deepslide
# multiprocessing

import os
os.add_dll_directory('path to openslide-win64-20171122/bin') #input openslide file

import sys
import multiprocessing as mp
from math import ceil
from os import listdir
from os.path import isfile, join
from PIL import Image
import openslide
import argparse




compression_factor = 1
Image.MAX_IMAGE_PIXELS = 1e10

from enum import Enum, IntEnum


class Axis(Enum):
    X = 1
    Y = 2


class SVSLevelRatio(IntEnum):
    LEVEL_0_BASE = 1
    LEVEL_1 = 4
    LEVEL_2 = 16
    LEVEL_3 = 32


class ResolutionLevel(IntEnum):
    LEVEL_0_BASE = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


def get_SVS_level_ratio(resolution_level):
    if resolution_level == ResolutionLevel.LEVEL_0_BASE:
        return SVSLevelRatio.LEVEL_0_BASE
    elif resolution_level == ResolutionLevel.LEVEL_1:
        return SVSLevelRatio.LEVEL_1
    elif resolution_level == ResolutionLevel.LEVEL_2:
        return SVSLevelRatio.LEVEL_2
    elif resolution_level == ResolutionLevel.LEVEL_3:
        return SVSLevelRatio.LEVEL_3


def get_start_positions(width, height, window_size, axis, overlapping_percentage):
    start_positions = []

    start_position = 0
    start_positions.append(start_position)

    dimension = width if axis == Axis.X else height

    while not (start_position + (window_size * (1 - overlapping_percentage))) > dimension:
        start_position = start_position + (window_size * (1 - overlapping_percentage))
        start_positions.append(int(start_position))

    return start_positions


def output_jpeg_tiles(full_image_path, full_output_path,
                      resolution_level,
                      overlapping_percentage,
                      window_size):  # converts svs image with meta data into just the jpeg image

    img = openslide.OpenSlide(full_image_path)
    width, height = img.level_dimensions[resolution_level]

    #   print("converting ", full_image_path, " with width ", width, ", height ", height, " and overlap ",
    #         overlapping_percentage)

    x_start_positions = get_start_positions(width, height, window_size, Axis.X, overlapping_percentage)
    y_start_positions = get_start_positions(width, height, window_size, Axis.Y, overlapping_percentage)

    #    print(x_start_positions)
    #    print(y_start_positions)

    total_number_of_patches = len(x_start_positions) * len(y_start_positions)
    tile_number = 1

    for x_index, x_start_position in enumerate(x_start_positions):
        for y_index, y_start_position in enumerate(y_start_positions):

            x_end_position = min(width, x_start_position + window_size)
            y_end_position = min(height, y_start_position + window_size)
            patch_width = x_end_position - x_start_position
            patch_height = y_end_position - y_start_position

            SVS_level_ratio = get_SVS_level_ratio(resolution_level)
            patch = img.read_region((x_start_position * SVS_level_ratio, y_start_position * SVS_level_ratio),
                                    resolution_level,
                                    (patch_width, patch_height))
            patch.load()
            patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
            patch_rgb.paste(patch, mask=patch.split()[3])

            print("\n")
            print("Patch data", x_start_position, y_start_position, resolution_level, patch_width, patch_height)
            print("Tile size for tile number " + str(tile_number) + ":" + str(patch.size))

            # compress the image
            #patch_rgb = patch_rgb.resize(
            #    (int(patch_rgb.size[0] / compression_factor), int(patch_rgb.size[1] / compression_factor)),
            #    Image.ANTIALIAS)


            # save the image
            output_subfolder = join(full_output_path, full_image_path.split('/')[-1][:-4])
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            output_image_name = join(output_subfolder,
                                     full_image_path.split('/')[-1][:-4] + '_' + str(x_index) + '_' + str(
                                         y_index) + '.jpg')
            # print(output_image_name)
            patch_rgb.save(output_image_name)
            print("Tile", tile_number, "/", total_number_of_patches, "created")
            tile_number = tile_number + 1





def data_process(input_folder_path,output_folder_path,start_at_image_name,resolution_level,overlap_percentage,window_size):



    input_folder_path = args.input_folder_path
    output_folder_path = args.output_folder_path
    start_at_image_name = args.start_at_image_name
    resolution_level = args.resolution_level
    overlapping_percentage = float("{0:.2f}".format(args.overlap_percentage / 100))
    window_size = args.window_size

    if not os.path.exists(input_folder_path):
        sys.exit("Error: Input folder doesn't exist")

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    image_names = [f for f in listdir(input_folder_path) if isfile(join(input_folder_path, f))]

    if '.DS_Store' in image_names:
        image_names.remove('.DS_Store')

    if start_at_image_name is not None:
        start = image_names.index(args.start_at)
        print("skipping the first", start)
        image_names = image_names[start + 2:]

    for image_name in image_names:
        full_image_path = input_folder_path + '/' + image_name
        output_path = output_folder_path + '/'
        output_jpeg_tiles(full_image_path, output_path, resolution_level, overlapping_percentage, window_size)


def multicore(input_folder_path,output_folder_path,start_at_image_name,resolution_level,overlap_percentage,window_size):
    
    pool = mp.Pool() 
    result = pool.map(data_process(input_folder_path,output_folder_path,start_at_image_name,resolution_level,overlap_percentage,window_size), range(10)) 
    #print(result) 





if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Split a WSI at a specific resolution in a .SVS file into .JPEG tiles.')
    parser.add_argument("-i", "--input_folder_path", type=str, help="The path to the input folder.", required=True)
    parser.add_argument("-o", "--output_folder_path", type=str, help="The path to the output folder."
                                                                    " If output folder doesn't exists at runtime "
                                                                    "the script will create it.",
                        required=True)
    parser.add_argument("-s", "--start_at_image_name", type=str, default=None, help="Resume from a certain filename."
                                                                                    " Default value is None.")
    parser.add_argument("-r", "--resolution_level", type=int, default=0, choices=[0, 1],
                        help="Resolution level for image to be split."
                            " Low level equals high resolution, lowest level is 0. Choose between {0, 1}."
                            " Default value is 0.")
    parser.add_argument("-op", "--overlap_percentage", type=int, default=0,
                        help="Overlapping percentage between patches."
                            " Default value is 0.")
    parser.add_argument("-ws", "--window_size", type=int, default=256,
                        help="Size for square window"
                            " Default value is 256.")

    args = parser.parse_args()




    multicore(args.input_folder_path,args.output_folder_path,args.start_at_image_name,args.resolution_level,args.overlap_percentage,args.window_size)
