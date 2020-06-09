import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from functools import partial
from stereovision.exceptions import BadBlockMatcherArgumentError

window_name = 'Tuner'
window_name2 = 'Disparity'


class Tuner:
    """
    The ``Tuner`` class discovers the ``BlockMatcher 's`` parameters and allows the user to adjust them online.
    """

    def __init__(self, img_left, img_right, block_matcher, settings_file, num_image, set_folder):
        self.imgL = img_left    # Undistorted and rectified left camera view image.
        self.imgR = img_right    # Undistorted and rectified right camera view image.
        self.block_matcher = block_matcher  # Selected blockmatcher
        self.pair = []  # Array to save image pair (ie, left and right image)
        self.pair.append(self.imgL)
        self.pair.append(self.imgR)
        self.num_image = num_image
        self.settings_folder = set_folder   # settings folder for each individual pair of images.

        self.shortest_dimension = min(self.pair[0].shape[:2])

        #: Settings chosen for ``BlockMatcher``

        """
            Load from settings file if it exists.
        """
        self.set_file = settings_file

        if os.path.isfile(self.set_file):   # Check if file exists before trying to load.
            block_matcher.load_settings(self.set_file)
        else:
            print('\nBlockmatcher settings file does not exist')
            print("\n")

        self.bm_settings = {}
        self.disparity = 0
        for parameter in self.block_matcher.parameter_maxima.keys():
            self.bm_settings[parameter] = []

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)

        # resize the window according to the screen resolution
        cv2.resizeWindow(window_name, 640, 400)
        cv2.resizeWindow(window_name2, 640, 380)

        self._initialize_trackbars()
        self.tune_pair(self.pair)

    def tune_pair(self, pair):
        """ Tune a pair of images to generate disparity."""

        self.pair = pair
        self.update_disparity_map()
        self._save_bm_state()

    def _set_value(self, parameter, new_value):
        """Try setting new parameter on ``block_matcher`` and update map."""

        print(parameter, new_value)
        try:
            self.block_matcher.__setattr__(parameter, new_value)
        except BadBlockMatcherArgumentError:
            return

        if parameter == 'SaveSett' and new_value == 1:
            # Save last blockmatcher settings after tuning.
            self.block_matcher.save_settings(self.num_image, self.settings_folder)
            print("\nSETTINGS SAVED... \n")

        self._update_trackbars(parameter)
        self.update_disparity_map()

    def _initialize_trackbars(self):
        """
        Initialize trackbars by discovering ``block_matcher``'s parameters.
        """
        for parameter in self.block_matcher.parameter_maxima.keys():
            maximum = self.block_matcher.parameter_maxima[parameter]
            if not maximum:
                maximum = self.shortest_dimension
            cv2.createTrackbar(parameter, window_name,
                               self.block_matcher.__getattribute__(parameter),
                               maximum, partial(self._set_value, parameter))

        # create switch for SAVE/NO SAVE SETTINGS functionality
        self.switch = 'SaveSett'
        cv2.createTrackbar(self.switch, window_name, 0, 1, partial(self._set_value, self.switch))

    def _update_trackbars(self, parameter):
        """
        Update trackbars by discovering newly set parameters and updating them.
        """

        maximum = self.block_matcher.parameter_maxima[parameter]
        if not maximum:
            maximum = self.shortest_dimension

        cv2.createTrackbar(parameter, window_name,
                           self.block_matcher.__getattribute__(parameter),
                           maximum, partial(self._set_value, parameter))

    def _save_bm_state(self):
        """ Save current state of 'block_matcher'."""

        for parameter in self.block_matcher.parameter_maxima.keys():
            self.bm_settings[parameter].append(
                self.block_matcher.__getattribute__(parameter))

    def update_disparity_map(self):
        """
        Update disparity map in GUI.
        The disparity image is normalized to the range 0-255 and then divided by
        255, because OpenCV multiplies it by 255 when displaying. This is
        because the pixels are stored as floating points.
        """

        disparity = self.block_matcher.get_disparity(self.pair)
        self.disparity = disparity
        norm_coeff = 255 / disparity.max()
        disparity = disparity * norm_coeff / 255

        cv2.imshow(window_name2, disparity)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def generate_disparity(self):
        """
        Calculate disparity map to be used to generate
        Point Cloud by reading last block matcher settings.
        """

        stereosgbm = cv2.StereoSGBM_create(minDisparity=self.bm_settings['minDisparity'][0],
                                           numDisparities=self.bm_settings['numDisparities'][0],
                                           blockSize=self.bm_settings['SADWindowSize'][0],
                                           P1=self.bm_settings['P1'][0],
                                           P2=self.bm_settings['P2'][0],
                                           disp12MaxDiff=self.bm_settings['disp12MaxDiff'][0],
                                           preFilterCap=2,
                                           uniquenessRatio=self.bm_settings['uniquenessRatio'][0],
                                           speckleWindowSize=self.bm_settings['speckleWindowSize'][0],
                                           speckleRange=self.bm_settings['speckleRange'][0],
                                           mode=self.bm_settings['fullDP'][0])

        img_l = self.pair[0]
        img_r = self.pair[1]

        disparity_map = stereosgbm.compute(img_l, img_r).astype(np.float32) / 16.0

        disparity_name = "disp_test_pair_" + str(self.num_image) + ".png"
        disparity_name = os.path.join(self.settings_folder, disparity_name)
        plt.imsave(disparity_name, disparity_map)

        print("\nSaving disparity map of test_pair... \n")
        print('%s saved' % disparity_name)
        print("\n")

        return disparity_map

    def point_cloud(self):
        """
        Generate 3D map/Point Cloud
        """

        print("\nGenerating the 3D map...")

        # Get new downscaled width and height

        img1 = self.pair[0]

        h, w = img1.shape[:2]

        # Focal length.
        focal_length = 688  # Value obtained from camera specifications.

        # Perspective transformation matrix
        # This transformation matrix is from the openCV documentation.
        Q = np.float32([[1, 0, 0, -0.5 * w],
                        [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                        [0, 0, 0, -focal_length],  # so that y-axis looks up.
                        [0, 0, 1, 0]])

        # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(self.disparity, Q)

        print("\n3D map generated")

        return points_3D, self.pair[0]


def object_id(disparity, ground_truth, points_3d, image):
    """
    Check for objects in ground truth image and classify based on colours.
    Each colour represents a class.
    """
    # (B - G - R)
    class1 = (255, 255, 255)
    class2 = (18, 91, 0)
    class3 = (184, 10, 7)
    class4 = (21, 241, 231)
    class5 = (0, 119, 245)
    class6 = (0, 97, 91)
    class7 = (177, 97, 202)
    class8 = (106, 145, 202)
    class9 = (0, 48, 82)
    class10 = (0, 250, 5)

    black = (0, 0, 0)
    white = (255, 255, 255)

    gt = cv2.imread(ground_truth)   # Ground truth image
    disparity_map = disparity    # Disparity image
    img_left = image    # Left camera image

    height, width = gt.shape[:2]    # Get height and width of ground truth image

    xy_points = np.zeros(gt.shape, dtype="float16")     # Create an array that will
    colours = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    colours_gt = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    for x in range(0, height):
        for y in range(0, width):
            if (gt[x, y] == class1).all():
                xy_points[x, y] = [x, y, 1]
                colours_gt[x, y] = class1

            elif (gt[x, y] == class2).all():
                xy_points[x, y] = [x, y, 2]
                colours_gt[x, y] = class2

            elif (gt[x, y] == class3).all():
                xy_points[x, y] = [x, y, 3]
                colours_gt[x, y] = class3

            elif (gt[x, y] == class4).all():
                xy_points[x, y] = [x, y, 4]
                colours_gt[x, y] = class4

            elif (gt[x, y] == class5).all():
                xy_points[x, y] = [x, y, 5]
                colours_gt[x, y] = class5

            elif (gt[x, y] == class6).all():
                xy_points[x, y] = [x, y, 6]
                colours_gt[x, y] = class6

            elif (gt[x, y] == class7).all():
                xy_points[x, y] = [x, y, 7]
                colours_gt[x, y] = class7

            elif (gt[x, y] == class8).all():
                xy_points[x, y] = [x, y, 8]
                colours_gt[x, y] = class8

            elif (gt[x, y] == class9).all():
                xy_points[x, y] = [x, y, 9]
                colours_gt[x, y] = class9

            elif (gt[x, y] == class10).all():
                xy_points[x, y] = [x, y, 10]
                colours_gt[x, y] = class10

            else:
                xy_points[x, y] = [x, y, 0]
                colours_gt[x, y] = black

    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()

    # Mask colours and 3D points.
    output_points = points_3d[mask_map]     # 3D (XYZ) points.
    output_class = xy_points[mask_map]     # Objects class in image (prints X, Y disparity points and class id)
    output_colours = colours[mask_map]      # BGR original colour of image.
    output_gt_colours = colours_gt[mask_map]     # Ground truth colours of objects in image.

    output_gray = cv2.cvtColor(colours, cv2.COLOR_RGB2GRAY)
    output_gray = output_gray[mask_map]     # Gray colour of image.

    return output_points, output_colours, output_gt_colours, output_gray, output_class
