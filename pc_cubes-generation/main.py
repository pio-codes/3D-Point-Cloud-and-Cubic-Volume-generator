# INITIALIZATION FROM TERMINAL
# python main.py --stereo_pair 'image folder path'/*.* 'bm settings folder' --output 'output folder path' --num_pair 'integer num'
# Example: python main.py --stereo_pair final_test/*.* final_test --output final_test --num_pair 2

import cv2
import numpy as np
from PIL import Image
import glob
import os
from argparse import ArgumentParser

from stereo_image_processing import StereoSGBM
from pointcloud_generation import Tuner, object_id
from cubes import cube, create_nii

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar red_gt
property uchar green_gt
property uchar blue_gt
property float gray
property float disparity_X
property float disparity_Y
property uchar class_id
end_header
'''


def read_folder(image_folder, num_pair):
    #   READ TEST IMAGES FROM FOLDER
    total_files = num_pair
    num_files = 0
    num_l = num_pair
    num_r = num_pair
    image_listLeft = []
    image_listRight = []

    while total_files > 0:
        for filename in glob.glob(image_folder):
            file = os.path.basename(filename)  # get filename from folder
            name, ext = file.split('.')  # separate name from ext
            if (name.startswith("l") or name.startswith("L")) and num_l > 0:
                iml = Image.open(filename)
                iml = np.asarray(iml)
                image_listLeft.append(iml)
                num_l = num_l - 1
                num_files = num_files + 1

            elif (name.startswith("r") or name.startswith("R")) and num_r > 0:
                imr = Image.open(filename)
                imr = np.asarray(imr)
                image_listRight.append(imr)
                num_r = num_r - 1
                num_files = num_files + 1

            if (num_l + num_r) == 0:
                total_files = 0

    return image_listLeft, image_listRight


def write_ply_all(fn, vertices, colours, colours_gt, colours_gray, obj_class):
    vertices = vertices.reshape(-1, 3)
    colours = colours.reshape(-1, 3)
    colours_gt = colours_gt.reshape(-1, 3)
    colours_gray = colours_gray.reshape(colours_gray.shape[0], 1)  # Horizontal to vertical reshaping.
    obj_class = obj_class.reshape(-1, 3)
    vertices = np.hstack([vertices, colours, colours_gt, colours_gray, obj_class])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(vertices))).encode('utf-8'))
        np.savetxt(f, vertices, fmt='%f %f %f %d %d %d %d %d %d %d %d %d %d')


def main():
    parser = ArgumentParser(description="Read images taken with stereo pair and use them to compute "
                                        "3D point clouds and ply ", )
    parser.add_argument("--stereo_pair", help="Path to folder with stereo pair images.")
    parser.add_argument("bm_settings", help="Path to block matcher's previous settings.")
    parser.add_argument("--output", help="Path to output file.")
    parser.add_argument("--num_pair", help="Number of image pairs to be analysed. Must be >= 1", type=int)

    args = parser.parse_args()

    image_folder = args.stereo_pair     # Folder containing image pairs.
    settings_folder = args.bm_settings   # Folder with previously saved settings.

    num_image = 0  # Number ID for each single image.
    num_pair = args.num_pair    # Number of image pairs to be analysed.

    print("\nReading and saving image files in folder... \n")
    left_images, right_images = read_folder(image_folder, num_pair)
    print("\nAll files read... \n")
    print("\n*=========================================================* \n")

    while num_pair > 0:
        print('START OF TEST PAIR ' + str(num_image) + '\n')

        img_left = left_images[num_image]
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right = right_images[num_image]
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # Reduce size of image to increase processing speed.
        h, w = img_left.shape[:2]
        if h >= 768 and w >= 1024:
            img_left = cv2.pyrDown(img_left)
            img_right = cv2.pyrDown(img_right)

        # Create folder to save disparity of each pair.
        test_folder = "test_pair_" + str(num_image)
        test_folder = os.path.join(settings_folder, test_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder, exist_ok=True)

        settings_file = "settings_" + str(num_image) + ".txt"
        bm_settings = os.path.join(test_folder, settings_file)  # Last block matcher settings/saved settings.

        gt_file = "gt_test_pair_" + str(num_image) + ".png"
        ground_truth = os.path.join(test_folder, gt_file)

        block_matcher = StereoSGBM()

        # Tune block matcher.
        tn = Tuner(img_left, img_right, block_matcher, bm_settings, num_image, test_folder)

        # Generate disparity image.
        disparity = tn.generate_disparity()

        # Generate 3D points.
        points, image_left = tn.point_cloud()

        # Classify objects and generate final values for each attribute of the image pair.
        output_points, output_colours, output_gt_colours, output_gray, output_class = \
            object_id(disparity, ground_truth, points, image_left)

        # Define name for output file and save to its own test folder.
        output_file = 'point_cloud_test_' + str(num_image) + '.ply'
        output_file = os.path.join(test_folder, output_file)

        # Generate point cloud file.
        print("\nCreating the output point cloud file of object classes... \n")
        write_ply_all(output_points, output_points, output_colours, output_gt_colours, output_gray, output_class)
        print('%s saved' % output_file)
        print("\n")

        # Generate cubes.
        print("\nGenerating 3D volumetric cubes... \n")
        cube_1, cube_2 = cube(output_points, output_gray, output_class, img_left)
        create_nii(cube_1, cube_2, test_folder)
        print("\nCubes generated and nii files saved... \n")

        print('\nEND OF TEST PAIR ' + str(num_image) + '\n')

        num_image = num_image + 1
        num_pair = num_pair - 1


if __name__ == "__main__":

    main()
