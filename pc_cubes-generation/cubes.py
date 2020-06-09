
import cv2
import numpy as np
import nibabel as nib
import os


def cube(points_3d, gray, class_id, img_left):

    # Loads the 3D point coordinates (X, Y, Z)
    xyz_load = points_3d

    height, width = xyz_load.shape[:2]

    im_height, im_width = img_left.shape[:2]
    im_depth = 100

    xyz_duplicate = np.zeros(xyz_load.shape, dtype="float16")
    xyz_aux = np.zeros(xyz_load.shape, dtype="float16")
    max_x = np.amax(xyz_load[:, 0])
    max_y = np.amax(xyz_load[:, 1])
    max_z = np.amax(xyz_load[:, 2])

    min_x = np.amin(xyz_load[:, 0])
    min_y = np.amin(xyz_load[:, 1])
    min_z = np.amin(xyz_load[:, 2])

    dif_x = max_x - min_x
    dif_y = max_y - min_y
    dif_z = max_z - min_z

    ratio_x = dif_x / im_width
    ratio_y = dif_y / im_height
    ratio_z = dif_z / im_depth

    # X, Y, Z Auxiliary
    for k in range(0, height):
        xyz_aux[k, 0] = xyz_load[k, 0] - min_x
        xyz_aux[k, 1] = xyz_load[k, 1] - min_y
        xyz_aux[k, 2] = xyz_load[k, 2] - min_z

    # X, Y, Z in rounded pixel values
    for i in range(0, height):
        xyz_duplicate[i, 0] = int(round(xyz_aux[i, 0] / ratio_x))
        xyz_duplicate[i, 1] = int(round(xyz_aux[i, 1] / ratio_y))
        xyz_duplicate[i, 2] = int(round(xyz_aux[i, 2] / ratio_z))

    # Save 'xyz_duplicate' as txt file by writing the array to disk
    with open('xyz_duplicate.txt', 'w') as outfile:
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(xyz_duplicate.shape))

        # Iterating through a n-dimensional array produces slices along the last axis. This is equivalent to data[i,:,:]
        for data_slice in xyz_duplicate:
            # The formatting string indicates that the values are written in left-justified columns
            # characters in width with 0 decimal places.
            data_slice = data_slice.reshape(-1, 3)
            np.savetxt(outfile, data_slice, fmt='%.f')

    cube_gray = np.zeros((im_width+1, im_height+1, 100+1), dtype=np.uint8)  # X, Y, Z
    cube_gt = np.zeros((im_width+1, im_height+1, 100+1), dtype=np.uint8)

    #   Generate cube for gray image
    for i in range(0, height):
        for k in range(0, int(xyz_duplicate[i, 2])+1):
            cube_gray[int(xyz_duplicate[i, 0]), int(xyz_duplicate[i, 1]), k] = gray[i]

    #   Generate cube for groundtruth image
    for i in range(0, height):
        if class_id[i, 2] == 1:
            for k in range(0, int(xyz_duplicate[i, 2])+1):
                cube_gt[int(xyz_duplicate[i, 0]), int(xyz_duplicate[i, 1]), k] = 255

    cube_gray = cv2.medianBlur(cube_gray, 5)
    cube_gt = cv2.medianBlur(cube_gt, 5)

    return cube_gray, cube_gt


def create_nii(cube_gray, cube_gt, output_folder):

    cubegray_nib = nib.Nifti1Image(cube_gray, affine=np.eye(4, 4))
    cubegt_nib = nib.Nifti1Image(cube_gt, affine=np.eye(4, 4))

    nib.save(cubegray_nib, os.path.join(output_folder, "gray.nii.gz"))
    nib.save(cubegt_nib, os.path.join(output_folder, "gt.nii.gz"))
