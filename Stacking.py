import os

import cv2  # библиотека работ с компьютерным зрением
import matplotlib.pylab as plt
import numpy as np  # библиотека работы с матрицами

from .KikuchiBands import KikuchiBands

# Script information for the file.
__author__ = "Maxim Osadchy (TG @MaxOsad)"
__version__ = "23.0419"
__date__ = "Apr 19, 2023"
__copyright__ = "Copyright (c) 2022 Maxim Osadchy"


class Stacking(KikuchiBands):
    number_of_stars = 3

    def __init__(self, file_name=None, im_gray=None):
        super().__init__(file_name, im_gray)

    def read_dataset_without_first(self, dataset_list, maxN=100):
        pattern1 = dataset_list[0]
        # pattern1.j = 1
        for j in range(2, maxN + 1):
            if pattern1.file_name[-5:] == ".tiff":
                fileName = "rim_" + f"{j:03}.tiff"
            else:
                fileName = str(j) + ".bmp"

            imagename = pattern1.dataset + "/" + fileName

            if not os.path.exists(imagename):
                continue

            if fileName == pattern1.file_name:
                # pattern1.j = j
                continue

            patternJ = Stacking(file_name=fileName)
            # patternJ.j = j

            dataset_list.append(patternJ)

    def set_keypoints_detector(self, detector_name=None):
        if detector_name is None:
            detector_name = self.detector_name.strip()
        else:
            detector_name = detector_name.strip()

        if detector_name == "ORB":
            detector = cv2.ORB_create()
        elif detector_name == "SIFT":
            detector = cv2.SIFT_create()

        self.detector = detector

        return detector

    def detectAndCompute(self, detector=None):
        if detector is None:
            if not hasattr(self, "detector"):
                self.set_keypoints_detector()
            detector = self.detector

        if not hasattr(self, "im_gray_ff"):
            self.im_gray_ff = self.first_filter(self.im_gray_f)

        self.keypoints, self.descriptors = detector.detectAndCompute(self.floaf_to_int255(self.im_gray_ff), None)

        return self.keypoints, self.descriptors

    def set_descriptorMatcher(self, descriptorMatcherName=None):
        if descriptorMatcherName is None:
            descriptorMatcherName = self.descriptor_matcher_name.strip()

        if descriptorMatcherName == "DESCRIPTOR_MATCHER_BRUTEFORCE":
            descriptorMatcherType = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE  # = 2
        elif descriptorMatcherName == "DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING":
            descriptorMatcherType = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING  # = 4
        elif descriptorMatcherName == "DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT":
            descriptorMatcherType = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT  # = 5
        elif descriptorMatcherName == "DESCRIPTOR_MATCHER_BRUTEFORCE_L1":
            descriptorMatcherType = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1  # = 3
        elif descriptorMatcherName == "DESCRIPTOR_MATCHER_BRUTEFORCE_SL2":
            descriptorMatcherType = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2  # = 6
        elif descriptorMatcherName == "DESCRIPTOR_MATCHER_FLANNBASED":
            descriptorMatcherType = cv2.DESCRIPTOR_MATCHER_FLANNBASED  # = 1

        self.descriptorMatcherType = descriptorMatcherType

        return descriptorMatcherType

    def set_findHomography_method(self, method_name=None):
        if method_name is None:
            method_name = self.method_name.strip()

        if method_name == "RANSAC":
            # RANSAC-based robust method
            method = cv2.RANSAC
        elif method_name == "LMEDS":
            # Least-Median robust method
            method = cv2.LMEDS
        elif method_name == "RHO":
            # PROSAC-based robust method
            method = cv2.RHO

        self.method = method

        return method

    def create_DescriptorMatcher(self):
        if not hasattr(self, "descriptorMatcherType"):
            self.set_descriptorMatcher()

        self.matcher = cv2.DescriptorMatcher_create(self.descriptorMatcherType)

        return self.matcher

    def match(self, descriptorsJ):
        # Creates a list of all matches, just like keypoints
        matches = self.matcher.match(
            np.rint(descriptorsJ).astype(np.uint8), np.rint(self.descriptors).astype(np.uint8), None
        )
        matches = sorted(matches, key=lambda x: x.distance)
        self.matches = matches

        return matches

    def warpPerspective(self, h, img=None):
        if img is None:
            img = self.im_gray_f.copy()

        # Applies a perspective transformation to an image.
        self.imRes_f = cv2.warpPerspective(img, h, (self.im_w, self.im_h))

        return self.imRes_f

    def dataset_warpPerspective(self, datasetList, detector_name=None, descriptor_matcher_name=None, method_name=None):
        if detector_name is not None:
            self.detector_name = detector_name.strip()

        if descriptor_matcher_name is not None:
            self.descriptor_matcher_name = descriptor_matcher_name.strip()

        if method_name is not None:
            self.method_name = method_name.strip()

        self.set_keypoints_detector()  # detector_name
        self.detectAndCompute()  # !
        self.set_descriptorMatcher()  # descriptorMatcherName
        self.set_findHomography_method()  # method_name
        self.create_DescriptorMatcher()

        jJ = []
        xX2 = []
        yY2 = []

        for patternJ in datasetList:
            if patternJ == self:
                self.imRes_f = self.im_gray_f
                # jJ.append(self.j)
                xX2.append(0)
                yY2.append(0)
                continue

            patternJ.detectAndCompute(self.detector)
            # Creates a list of all matches, just like keypoints
            matches = self.match(patternJ.descriptors)
            # good = matches[:]
            # Prints empty array of size equal to (matches, 2)
            pointsJ = np.zeros((len(matches), 2), dtype=np.float32)
            pointsTrain = np.zeros((len(matches), 2), dtype=np.float32)

            for ii, match in enumerate(matches):
                # gives index of the descriptor in the list of query descriptors
                pointsJ[ii, :] = patternJ.keypoints[match.queryIdx].pt
                # gives index of the descriptor in the list of train descriptors
                pointsTrain[ii, :] = self.keypoints[match.trainIdx].pt

            patternJ.h, mask = cv2.findHomography(pointsJ, pointsTrain, self.method)

            # matchesMask = mask.ravel().tolist()  # ?

            # Applies a perspective transformation to an image.
            patternJ.warpPerspective(patternJ.h)

            # jJ.append(patternJ.j)
            xX2.append(-patternJ.h[0, 2])
            yY2.append(-patternJ.h[0, 2])

        return xX2, yY2, jJ

    def dataset_full_scan(self, datasetList):
        jJ = []
        xX1 = []
        yY1 = []

        frame = self.frame

        im_w = self.im_w
        im_h = self.im_h

        if not hasattr(self, "im_gray_ff"):
            self.im_gray_ff = self.first_filter(self.im_gray_f)

        imTrainFrame = self.im_gray_ff[frame : im_h - frame, frame : im_w - frame]

        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        for patternJ in datasetList:
            patternJ.h = h.copy()
            if patternJ == self:
                self.imRes_f = self.im_gray_f
                # jJ.append(self.j)
                xX1.append(0)
                yY1.append(0)
                continue

            if not hasattr(patternJ, "im_gray_ff"):
                patternJ.im_gray_ff = patternJ.first_filter(patternJ.im_gray_f)

            imFrame = patternJ.im_gray_ff[frame : im_h - frame, frame : im_w - frame]
            minSum = np.sum(np.square(np.subtract(imTrainFrame, imFrame)))

            ix = 0
            iy = 0

            for dx in range(-self.apertura, self.apertura + 1):
                for dy in range(-self.apertura, self.apertura + 1):
                    imFrame = patternJ.im_gray_ff[dy + frame : dy + im_h - frame, dx + frame : dx + im_w - frame]
                    sum_SQ = np.sum(np.square(imTrainFrame - imFrame))

                    if minSum >= sum_SQ:
                        if minSum > sum_SQ:
                            minSum = sum_SQ
                            ix = dx
                            iy = dy
                        elif (abs(dx) + abs(dy)) < (abs(ix) + abs(iy)):
                            minSum = sum_SQ
                            ix = dx
                            iy = dy

            xX1.append(ix)
            yY1.append(-iy)
            # jJ.append(patternJ.j)

            patternJ.h[0, 2] = ix
            patternJ.h[1, 2] = iy

            patternJ.imRes_f = cv2.warpAffine(
                patternJ.im_gray_f, patternJ.h, (im_w, im_h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )

        return xX1, yY1, jJ

    def dataset_correlation(self, datasetList):
        if self.show_process:
            print("dataset_correlation")

        jJ = []
        xX1 = []
        yY1 = []

        frame = self.frame

        im_w = self.im_w
        im_h = self.im_h

        if not hasattr(self, "im_gray_ff"):
            self.im_gray_ff = self.first_filter(self.im_gray_f)

        imTrainFrame = self.im_gray_ff[frame : im_h - frame, frame : im_w - frame]

        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        for patternJ in datasetList:
            patternJ.h = h.copy()

            if patternJ == self:
                self.imRes_f = self.im_gray_f
                # jJ.append(self.j)
                xX1.append(0)
                yY1.append(0)
                continue

            if not hasattr(patternJ, "im_gray_ff"):
                patternJ.im_gray_ff = patternJ.first_filter(patternJ.im_gray_f)

            imFrame = patternJ.im_gray_ff[frame : im_h - frame, frame : im_w - frame]
            maxSum = (imTrainFrame * imFrame).sum()

            ix = 0
            iy = 0

            for dx in range(-self.apertura, self.apertura + 1):
                for dy in range(-self.apertura, self.apertura + 1):
                    imFrame = patternJ.im_gray_ff[dy + frame : dy + im_h - frame, dx + frame : dx + im_w - frame]
                    sum_SQ = (imTrainFrame * imFrame).sum()

                    if maxSum < sum_SQ:
                        maxSum = sum_SQ
                        ix = dx
                        iy = dy

            xX1.append(ix)
            yY1.append(-iy)
            # jJ.append(patternJ.j)

            patternJ.h[0, 2] = ix
            patternJ.h[1, 2] = iy

            if self.show_process:
                print(patternJ.file_name, ix, iy)

            patternJ.imRes_f = cv2.warpAffine(
                patternJ.imGray_f, patternJ.h, (im_w, im_h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )

        return xX1, yY1, jJ

    def dataset_stars_center(self, datasetList):
        if self.show_process:
            print("dataset_stars_center")

        jJ = []
        xX1 = []
        yY1 = []

        self.Filter_stars = "GaussianBlur"
        self.paramFilter_stars, self.threshold_level_stars = self.choice_stars_filter_param(filt=self.Filter_stars)

        self.im_gray_ff = self.first_filter(filt=self.Filter_stars, param1=self.paramFilter_stars)

        thresh = self.threshold(threshold_level=self.threshold_level_stars)
        self.findContours(thresh=thresh)
        self.moments()

        # print("  ", self.j, self.starsCenters_x, self.starsCenters_y)

        # [x,y]
        pointsTrain = np.array(
            [
                [self.starsCenters_x[0], self.starsCenters_y[0]],
                [self.starsCenters_x[1], self.starsCenters_y[1]],
                [self.starsCenters_x[2], self.starsCenters_y[2]],
            ]
        ).astype(np.float32)

        for patternJ in datasetList:
            if patternJ == self:
                self.imRes_f = self.im_gray_f
                # jJ.append(self.j)
                xX1.append(0)
                yY1.append(0)
                continue

            # patternJ.paramFilter_stars, patternJ.threshold_level_stars =
            # patternJ.choice_stars_filter_param(filt=self.Filter_stars)

            patternJ.im_gray_ff = patternJ.first_filter(filt=self.Filter_stars, param1=self.paramFilter_stars)

            patternJ.threshold_level_stars = self.threshold_level_stars
            thresh = patternJ.threshold(threshold_level=self.threshold_level_stars)
            patternJ.findContours(thresh=thresh)
            patternJ.moments()

            # print("  ", patternJ.j, patternJ.starsCenters_x, patternJ.starsCenters_y)

            pointsJ = np.array(
                [
                    [patternJ.starsCenters_x[0], patternJ.starsCenters_y[0]],
                    [patternJ.starsCenters_x[1], patternJ.starsCenters_y[1]],
                    [patternJ.starsCenters_x[2], patternJ.starsCenters_y[2]],
                ]
            ).astype(np.float32)

            # print("sum", patternJ.j, np.sum(abs(pointsTrain - pointsJ)))

            if np.sum(abs(pointsTrain - pointsJ)) > 100:
                contours = patternJ.contours
                len_contours = len(contours)
                patternJ.moments(number_of_stars=len_contours)  # def moments(self, img=None, number_of_stars=None):

                starsCenters_x = []
                starsCenters_y = []
                for i_star in range(3):
                    i_min = i_star
                    sum_min = abs(self.starsCenters_x[i_star] - patternJ.starsCenters_x[i_star]) + abs(
                        self.starsCenters_y[i_star] - patternJ.starsCenters_y[i_star]
                    )

                    for i in range(len_contours):
                        sum_star = abs(self.starsCenters_x[i_star] - patternJ.starsCenters_x[i]) + abs(
                            self.starsCenters_y[i_star] - patternJ.starsCenters_y[i]
                        )
                        if sum_star < sum_min:
                            sum_min = sum_star
                            i_min = i

                    starsCenters_x.append(patternJ.starsCenters_x[i_min])
                    starsCenters_y.append(patternJ.starsCenters_y[i_min])

                pointsJ_tmp = np.array(
                    [
                        [starsCenters_x[0], starsCenters_y[0]],
                        [starsCenters_x[1], starsCenters_y[1]],
                        [starsCenters_x[2], starsCenters_y[2]],
                    ]
                ).astype(np.float32)

                if np.sum(abs(pointsTrain - pointsJ_tmp)) < 100:
                    patternJ.starsCenters_x = starsCenters_x
                    patternJ.starsCenters_y = starsCenters_y

                    pointsJ = np.array(
                        [
                            [patternJ.starsCenters_x[0], patternJ.starsCenters_y[0]],
                            [patternJ.starsCenters_x[1], patternJ.starsCenters_y[1]],
                            [patternJ.starsCenters_x[2], patternJ.starsCenters_y[2]],
                        ]
                    ).astype(np.float32)
                else:
                    pass
                    # print("XXXXXXXXXXXXXXXXXX")

                # print("sum =========== ", patternJ.j, np.sum(abs(pointsTrain - pointsJ)))

            patternJ.h = cv2.getAffineTransform(pointsJ, pointsTrain)

            # xX2.append(-h[0,2])

            ix = patternJ.h[0, 2]
            iy = patternJ.h[1, 2]

            xX1.append(ix)
            yY1.append(-iy)
            # jJ.append(patternJ.j)

            if self.show_process:
                print(patternJ.file_name, ix, iy)

            # print(patternJ.fileName, patternJ.j, ix, iy)

            patternJ.imRes_f = cv2.warpAffine(
                patternJ.im_gray_f,
                patternJ.h,
                (patternJ.im_w, patternJ.im_h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
            pass

        return xX1, yY1, jJ

    def dataset_transformECC(self, datasetList):
        if self.show_process:
            print("dataset_transformECC")

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 50

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        if not hasattr(self, "im_gray_ff"):
            self.im_gray_ff = self.first_filter(self.im_gray_f)

        for patternJ in datasetList:
            patternJ.h = warp_matrix.copy()
            if not hasattr(patternJ, "im_gray_ff"):
                patternJ.im_gray_ff = patternJ.first_filter(patternJ.im_gray_f)

            # Run the ECC algorithm. The results are stored in warp_matrix.
            try:
                _, patternJ.h = cv2.findTransformECC(
                    self.im_gray_ff, patternJ.im_gray_ff, patternJ.h, self.warp_mode, criteria, None, 5
                )
            except cv2.CV_Error:
                pass
                print("Warning: find transform failed. Set warp as identity")

            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                # Applies a perspective transformation to an image.
                patternJ.warpPerspective(patternJ.h)
                # patternJ.imRes_f = cv2.warpPerspective(patternJ.imGray_f, h, (self.im_w,self.im_h),
                # flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                patternJ.imRes_f = cv2.warpAffine(
                    patternJ.im_gray_f,
                    patternJ.h,
                    (self.im_w, self.im_h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                )
            patternJ.h = warp_matrix

    # stacking
    def stack(self, datasetList):
        if hasattr(self, "apertura"):
            apertura = self.apertura
        else:
            apertura = 6
            Stacking.apertura = apertura

        stack = 0.0
        stackMask = 0.000001
        a1 = apertura + 1

        for patternJ in datasetList:
            if hasattr(patternJ, "h"):
                # print ("--------", patternJ.j, np.sum(abs(patternJ.h)))
                print("--------", np.sum(abs(patternJ.h)))
                if np.sum(abs(patternJ.h)) > 25:
                    continue

            imRes_f = self.float255(patternJ.imRes_f)

            stack += imRes_f

            maxval = imRes_f.max()
            _, thresh = cv2.threshold(imRes_f, 0.9, maxval, cv2.THRESH_BINARY)  # ?
            thresh[a1 : patternJ.im_h - a1, a1 : patternJ.im_w - a1] = maxval
            stackMask += thresh / maxval

        stack /= stackMask
        stack = self.floaf_to_int255(stack)

        coord = []
        for pattern in datasetList:
            if hasattr(pattern, "h"):
                if hasattr(pattern, "id"):
                    coord.append((pattern.id, pattern.h.tolist()))

        return stack, coord

    def set_dataset_patterns_bsbc(self, datasetList):
        pattern1 = datasetList[0]
        patternMaxBS = pattern1
        # bs_maxJ = 1

        values_bandSlope = np.array(range(len(datasetList)), dtype="f")
        values_bandContrast = np.array(range(len(datasetList)), dtype="f")

        i = 0
        for patternJ in datasetList:
            if not hasattr(patternJ, "bsbc_rotation_matrix"):
                Stacking.bsbc_rotation_matrix = patternJ.get_rotation_matrix()
                Stacking.bsbc_rotation_center = patternJ.bsbc_rotation_center

            patternJ.get_bs_bc()

            values_bandSlope[i] = patternJ.bandSlope
            values_bandContrast[i] = patternJ.bandContrast
            i += 1

            if patternJ.bandSlope > patternMaxBS.bandSlope:
                patternMaxBS = patternJ
                # bs_maxJ = patternJ.j

        return values_bandSlope, values_bandContrast

    def stack_quality_analysis(self, stack_img=None, datasetList=None):
        pattern1 = datasetList[0]

        if hasattr(pattern1, "averaging"):
            averaging = pattern1.averaging
        else:
            averaging = 120
            Stacking.averaging = averaging

        if hasattr(pattern1, "averaging_position"):
            averaging_position = pattern1.averaging_position
        else:
            averaging_position = pattern1.im_w // 2 + 40
            Stacking.averaging_position = averaging_position

        if datasetList is not None:
            values_bandSlope, values_bandContrast = pattern1.set_dataset_patterns_bsbc(datasetList)
            plt.plot(values_bandContrast, values_bandSlope, "b.", label="patterns")

        if stack_img is not None:
            patternStack = Stacking(im_gray=stack_img)
            patternStack.get_bs_bc()
            plt.plot(patternStack.bandContrast, patternStack.bandSlope, "r*", markersize=15, label="stack")

        plt.xlabel("bandContrast (BC)")
        plt.ylabel("bandSlope (BS)")
        plt.legend(loc="upper left", fancybox=True, shadow=True)
        plt.grid(True)

        # fileName = "bc_bs_"+str(averaging)+".png"
        # plt.savefig(fileName, dpi=200)

        return plt
