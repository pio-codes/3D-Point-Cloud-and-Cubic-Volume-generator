import cv2
import numpy as np
import simplejson
import os

from stereovision.exceptions import ChessboardNotFoundError
from stereovision.exceptions import (InvalidSearchRangeError,
                                     InvalidWindowSizeError,
                                     InvalidBMPresetError,
                                     InvalidNumDisparitiesError,
                                     InvalidSADWindowSizeError,
                                     InvalidUniquenessRatioError,
                                     InvalidSpeckleWindowSizeError,
                                     InvalidSpeckleRangeError,
                                     InvalidFirstDisparityChangePenaltyError,
                                     InvalidSecondDisparityChangePenaltyError)


class BlockMatcher(object):

    """
    Block matching algorithms.
    This abstract class exposes the interface for subclasses that wrap OpenCV's
    block matching algorithms. Doing so makes it possible to use them in the
    strategy pattern.
    Each ``BlockMatcher`` protects its block matcher's parameters by using
    getters and setters. It exposes its settable parameter and their maximum
    values, if they exist, in the dictionary ``parameter_maxima``.
    ``load_settings``, ``save_settings`` and ``get_3d`` are implemented on
    ``BlockMatcher`` itself, as these are independent of the block matching
    algorithm. Subclasses are expected to implement ``_replace_bm`` and
    ``get_disparity``, as well as the getters and setters. They are also
    expected to call ``BlockMatcher``'s ``__init__`` after setting their own
    private variables.
    """

    #: Dictionary of parameter names associated with their maximum values
    parameter_maxima = {}

    def __init__(self, settings=None):
        """Set block matcher parameters and load from file if necessary."""
        #: Block matcher object used for computing point clouds
        self._block_matcher = None
        self._replace_bm()
        if settings:
            self.load_settings(settings)

    def load_settings(self, settings):
        """Load settings from file"""
        with open(settings) as settings_file:
            settings_dict = simplejson.load(settings_file)
        for key, value in settings_dict.items():
            self.__setattr__(key, value)

    def save_settings(self, num_image, save_path):
        """Save block matcher settings to a file object"""
        settings = {}
        settings_file = "settings_" + str(num_image) + ".txt"
        settings_file = os.path.join(save_path, settings_file)  # Path and file joined to save in specific folder.
        for parameter in self.parameter_maxima:
            settings[parameter] = self.__getattribute__(parameter)
        with open(settings_file, "w") as settings_file:
            simplejson.dump(settings, settings_file, indent=1)

    def _replace_bm(self):
        """Replace block matcher with new parameters"""
        raise NotImplementedError


class StereoSGBM(BlockMatcher):

    """A semi-global block matcher."""

    parameter_maxima = {"minDisparity": None,
                        "numDisparities": None,
                        "SADWindowSize": 11,
                        "P1": None,
                        "P2": None,
                        "disp12MaxDiff": None,
                        "uniquenessRatio": 15,
                        "speckleWindowSize": 200,
                        "speckleRange": 2,
                        "fullDP": 1,
                        "SaveSett": 1}

    @property
    def minDisparity(self):
        """Return private ``_min_disparity`` value."""
        return self._min_disparity

    @minDisparity.setter
    def minDisparity(self, value):
        """Set private ``_min_disparity`` and reset ``_block_matcher``."""
        self._min_disparity = value
        self._replace_bm()

    @property
    def numDisparities(self):
        """Return private ``_num_disp`` value."""
        return self._num_disp

    @numDisparities.setter
    def numDisparities(self, value):
        """Set private ``_num_disp`` and reset ``_block_matcher``."""
        if value > 0 and value % 16 == 0:
            self._num_disp = value
        else:
            raise InvalidNumDisparitiesError("numDisparities must be a "
                                             "positive integer evenly "
                                             "divisible by 16.")
        self._replace_bm()

    @property
    def SADWindowSize(self):
        """Return private ``_sad_window_size`` value."""
        return self._sad_window_size

    @SADWindowSize.setter
    def SADWindowSize(self, value):
        """Set private ``_sad_window_size`` and reset ``_block_matcher``."""
        if 1 <= value <= 11 and value % 2:
            self._sad_window_size = value
        else:
            raise InvalidSADWindowSizeError("SADWindowSize must be odd and "
                                            "between 1 and 11.")
        self._replace_bm()

    @property
    def uniquenessRatio(self):
        """Return private ``_uniqueness`` value."""
        return self._uniqueness

    @uniquenessRatio.setter
    def uniquenessRatio(self, value):
        """Set private ``_uniqueness`` and reset ``_block_matcher``."""
        if 5 <= value <= 15:
            self._uniqueness = value
        else:
            raise InvalidUniquenessRatioError("Uniqueness ratio must be "
                                              "between 5 and 15.")
        self._replace_bm()

    @property
    def speckleWindowSize(self):
        """Return private ``_speckle_window_size`` value."""
        return self._speckle_window_size

    @speckleWindowSize.setter
    def speckleWindowSize(self, value):
        """Set private ``_speckle_window_size`` and reset ``_block_matcher``."""
        if 0 <= value <= 200:
            self._speckle_window_size = value
        else:
            raise InvalidSpeckleWindowSizeError("Speckle window size must be 0 "
                                                "for disabled checks or "
                                                "between 50 and 200.")
        self._replace_bm()

    @property
    def speckleRange(self):
        """Return private ``_speckle_range`` value."""
        return self._speckle_range

    @speckleRange.setter
    def speckleRange(self, value):
        """Set private ``_speckle_range`` and reset ``_block_matcher``."""
        if value >= 0:
            self._speckle_range = value
        else:
            raise InvalidSpeckleRangeError("Speckle range cannot be negative.")
        self._replace_bm()

    @property
    def disp12MaxDiff(self):
        """Return private ``_max_disparity`` value."""
        return self._max_disparity

    @disp12MaxDiff.setter
    def disp12MaxDiff(self, value):
        """Set private ``_max_disparity`` and reset ``_block_matcher``."""
        self._max_disparity = value
        self._replace_bm()

    @property
    def P1(self):
        """Return private ``_P1`` value."""
        return self._P1

    @P1.setter
    def P1(self, value):
        """Set private ``_P1`` and reset ``_block_matcher``."""
        if value < self.P2:
            self._P1 = value
        else:
            raise InvalidFirstDisparityChangePenaltyError("P1 must be less "
                                                          "than P2.")
        self._replace_bm()

    @property
    def P2(self):
        """Return private ``_P2`` value."""
        return self._P2

    @P2.setter
    def P2(self, value):
        """Set private ``_P2`` and reset ``_block_matcher``."""
        if value > self.P1:
            self._P2 = value
        else:
            raise InvalidSecondDisparityChangePenaltyError("P2 must be greater "
                                                           "than P1.")
        self._replace_bm()

    @property
    def fullDP(self):
        """Return private ``_full_dp`` value."""
        return self._full_dp

    @fullDP.setter
    def fullDP(self, value):
        """Set private ``_full_dp`` and reset ``_block_matcher``."""
        self._full_dp = bool(value)
        self._replace_bm()

    @property
    def SaveSett(self):
        """Return private ``_save_set`` value."""
        return self._save_set

    @SaveSett.setter
    def SaveSett(self, value):
        """Set private ``_save_set`` and reset ``_block_matcher``."""
        self._save_set = bool(value)
        self._replace_bm()

    def _replace_bm(self):
        """Replace ``_block_matcher`` with current values."""
        self._block_matcher = cv2.StereoSGBM_create(minDisparity=self._min_disparity,
                                                    numDisparities=self._num_disp,
                                                    blockSize=self._sad_window_size,
                                                    P1=self._P1,
                                                    P2=self._P2,
                                                    disp12MaxDiff=self._max_disparity,
                                                    preFilterCap=2,
                                                    uniquenessRatio=self._uniqueness,
                                                    speckleWindowSize=self._speckle_window_size,
                                                    speckleRange=self._speckle_range,
                                                    mode=self._full_dp)

    def __init__(self, min_disparity=1, num_disp=96, sad_window_size=3,
                 uniqueness=5, speckle_window_size=200, speckle_range=32,
                 p1=216, p2=864, max_disparity=8, full_dp=True, settings=None, save_set=0):
        """Instantiate private variables and call superclass initializer."""
        #: Minimum number of disparities. Normally 0, can be adjusted as needed
        self._min_disparity = min_disparity
        #: Number of disparities
        self._num_disp = num_disp
        #: Matched block size
        self._sad_window_size = sad_window_size
        #: Uniqueness ratio for found matches
        self._uniqueness = uniqueness
        #: Maximum size of smooth disparity regions to invalid by noise
        self._speckle_window_size = speckle_window_size
        #: Maximum disparity range within connected component
        self._speckle_range = speckle_range
        #: Penalty on disparity change by +-1 between neighbor pixels
        self._P1 = p1
        #: Penalty on disparity change by multiple neighbour pixels
        self._P2 = p2
        #: Maximum left-right disparity. 0 to disable check
        self._max_disparity = max_disparity
        #: Boolean to use full-scale two-pass dynamic algorithm
        self._full_dp = full_dp
        #: Boolean to decide if settings are to be saved
        self._save_set = save_set
        #: StereoSGBM whose state is controlled
        self._block_matcher = cv2.StereoSGBM()
        super(StereoSGBM, self).__init__(settings)

    def get_disparity(self, pair):
        """Compute disparity from image pair (left, right)."""
        return self._block_matcher.compute(pair[0], pair[1]).astype(np.float32) / 16.0

