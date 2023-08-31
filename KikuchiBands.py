import math

import cv2  # библиотека работ с компьютерным зрением
import matplotlib.pylab as plt
import numpy as np  # библиотека работы с матрицами

# from PIL import Image

# Script information for the file.
__author__ = "Maxim Osadchy (TG @MaxOsad)"
__version__ = "23.0419"
__date__ = "Apr 19, 2023"
__copyright__ = "Copyright (c) 2022 Maxim Osadchy"


class KikuchiBands:
    isColor = 0
    video = False
    show_process = False  # troubleshooting system

    def __init__(self, file_name=None, im_gray=None):
        self.file_name = file_name

        if file_name is not None:
            img = self.imread()
            self.im_gray = self.cut_footer(img)

        if im_gray is not None:
            if im_gray.dtype.name == "uint8":
                self.im_gray = self.cut_footer(im_gray)
                self.im_gray_f = self.float255(self.im_gray)  # float
            elif im_gray.dtype.name == "float32":
                self.im_gray_f = im_gray
                self.im_gray = self.floaf_to_int255(im_gray)  # float
        else:
            self.im_gray_f = self.float255(self.im_gray)  # float

        self.feedBack = "@Maxosad"

    def cut_footer(self, img):
        im_w = img.shape[1]  # ширина
        im_h = img.shape[0]  # высота

        for i in range(im_h):
            if img[im_h - i - 1 : im_h - i, 0:6].max() > 0:
                break
            if img[im_h - i - 1 : im_h - i, im_h - 5 : im_h].max() > 0:
                break

        return img[0 : im_h - i, 0:im_w]

    def imread(self):
        if self.dataset:
            dir_ = self.dataset + "/"
        else:
            dir_ = ""

        imagename = dir_ + self.file_name
        imJ = cv2.imread(imagename, 0)

        # Gray
        if self.isColor == 0:
            im_gray = imJ
        else:
            im_gray = cv2.cvtColor(imJ, cv2.COLOR_BGR2GRAY)

        if self.show_process:
            img = im_gray
            title = ""

            pic_box = plt.figure(figsize=(22, 8))

            pic_box.add_subplot(121).set_title(title + " shape=" + str(img.shape))
            plt.imshow(img, cmap="gray")

            pic_box.add_subplot(122).set_title(
                "dtype=" + str(img.dtype.name) + " min=" + str(img.min()) + " max=" + str(img.max())
            )
            plt.hist(img.ravel(), bins=256)

            plt.show()

        return im_gray

    def def_imShow(self, img, img2=None, title1="", title2="", cmap="gray"):
        if self.show_process:
            print("def_imShow")

        if img2 is not None:
            pic_box = plt.figure(figsize=(22, 8))
            pic_box.add_subplot(122).set_title(
                title2
                + " shape="
                + str(img.shape)
                + " dtype="
                + str(img.dtype.name)
                + " min="
                + str(img.min())
                + " max="
                + str(img.max())
            )
            plt.imshow(img2, cmap="gray")
            pic_box.add_subplot(121).set_title(
                title1
                + " shape="
                + str(img.shape)
                + " dtype="
                + str(img.dtype.name)
                + " min="
                + str(img.min())
                + " max="
                + str(img.max())
            )
        else:
            pic_box = plt.figure(figsize=(20, 16))
            pic_box.add_subplot(111).set_title(
                title1
                + " shape="
                + str(img.shape)
                + " dtype="
                + str(img.dtype.name)
                + " min="
                + str(img.min())
                + " max="
                + str(img.max())
            )

        plt.imshow(img, cmap="gray")
        plt.show()

    def float255(self, img):
        img = np.float32(img)
        img = img - img.min()
        img = 255.0 * img / img.max()

        return img

    def first_filter(self, img=None, filt=None, param1=None, param2=None, param3=None):
        if self.show_process:
            print("first_filter")

        if img is None:
            img = self.im_gray_f

        if filt is None:
            filt = self.filter.strip()
        else:
            filt = filt.strip()

        if param1 is None:
            param1 = self.param_filter
        if param2 is None:
            param2 = param1 * 2
        if param3 is None:
            param3 = param1 / 2

        if filt == "blur":
            res = cv2.blur(img, (param1, param1))
            # self.filter = filt
            # self.param_filter = param1
        elif filt == "bilateralfilter":
            res = cv2.bilateralfilter(img, param1, param2, param3)  # imGray = cv2.bilateralfilter(imGray,9,75,75)
            # self.filter = filt
            # self.param_filter = param1
        elif filt == "GaussianBlur":
            res = cv2.GaussianBlur(img, (param1, param1), 0)  # <50
            # self.filter = filt
            # self.param_filter = param1
        elif filt == "medianBlur":
            res = cv2.medianBlur(self.floaf_to_int255(img), param1)  # 5
            # self.filter = filt
            # self.param_filter = param1
        else:
            res = img.copy()

        res = self.float255(res)

        if self.show_process:
            self.def_imShow(img, res)

        return res

    def floaf_to_int255(self, img):
        # конвертируем float в int с максимумом в 255
        im_ = img.copy()
        im_ = np.float32(im_)
        im_ = im_ - im_.min()
        im_ = 255.0 * im_ / im_.max()
        im_ = np.rint(im_)  # округление

        return im_.astype(np.uint8)

    def choice_stars_filter_param(self, filt=None, threshold_level=None):
        if self.show_process:
            print("choice_stars_filter_param")

        if filt is None:
            filt = self.filter_stars

        param_filter = 3
        number_of_stars = self.number_of_stars

        im_gray_ff = self.first_filter(filt=filt, param1=param_filter)
        img = self.floaf_to_int255(im_gray_ff)
        img_max = img.max()

        level_1 = (self.im_w * self.im_h) // 3000
        ns_1 = number_of_stars - 1

        for i in range(200 + 1):
            threshold_level = i
            _, thresh = cv2.threshold(img, img_max - threshold_level, img_max, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )  # contours, hierarchy = cv2.findContours(thresh, 1, 2)

            len_contours = len(contours)

            if len_contours >= number_of_stars:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                max_contourArea = cv2.contourArea(sorted_contours[ns_1])
            else:
                max_contourArea = 0

            if max_contourArea > level_1:
                break

        len_contours_old = None

        for i in range(200 + 1):
            param_filter = 1 + 2 * i
            im_gray_ff = self.first_filter(filt=filt, param1=param_filter)

            img = self.floaf_to_int255(im_gray_ff)
            img_max = img.max()

            _, thresh = cv2.threshold(img, img_max - threshold_level, img_max, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )  # contours, hierarchy = cv2.findContours(thresh, 1, 2)

            len_contours = len(contours)

            if len_contours_old is not None:
                if len_contours_old == len_contours:
                    continue
                elif len_contours_old < len_contours:
                    # i -= 1
                    break

            if len_contours < number_of_stars:  # if len_contours <= 3 + 1:
                break

            i_old = i
            if len_contours == number_of_stars:  # if len_contours <= 3 + 1:
                break

            len_contours_old = len_contours

        self.param_filter_stars = 1 + 2 * i_old
        self.threshold_level_stars = threshold_level

        return self.param_filter_stars, self.threshold_level_stars

    # threshold STRONG
    def threshold(self, threshold_level=None):
        if self.show_process:
            print("threshold")

        if threshold_level is None:
            threshold_level = self.threshold_level_stars

        img = self.floaf_to_int255(self.im_gray_ff)

        img_max = img.max()

        _, thresh = cv2.threshold(
            img, img_max - threshold_level, img_max, cv2.THRESH_BINARY
        )  # _, thresh = cv2.threshold(im_gray_ff,a-10, a, cv2.THRESH_BINARY_INV)

        self.threshold_img = thresh

        if self.show_process:
            print("threshold_level=", threshold_level)
            self.def_imShow(thresh)

        return thresh

    def findContours(self, thresh=None):
        if self.show_process:
            print("findContours")

        if thresh is None:
            if not hasattr(self, "threshold_img"):
                thresh = self.threshold()
            thresh = self.threshold_img

        self.contours, _ = cv2.findContours(
            thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )  # contours, hierarchy = cv2.findContours(thresh, 1, 2)

        if self.video or self.show_process:
            len_contours = len(self.contours)

            img = self.im_gray.copy()

            color = (0, 0, 0)
            for i in range(len_contours):
                # Draw the contour
                cv2.drawContours(img, self.contours, i, color, 2)

            if self.show_process:
                self.def_imShow(img)

            return img
        return thresh

    # adaptiv
    def threshold_plus_findContours(self):
        if self.show_process:
            print("threshold_plus_findContours")

        number_of_stars = self.number_of_stars

        if not hasattr(self, "im_gray_ff"):
            self.im_gray_ff = self.first_filter()

        threshold_img = self.threshold()
        contours, hierarchy = cv2.findContours(threshold_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        len_contours = len(contours)
        if len_contours >= number_of_stars:
            self.contours = contours

            return threshold_img

        threshold_level_good = self.threshold_level_stars
        self.threshold_level_stars += 1

        i = 0
        len_contours_good = 0

        while True:
            threshold_img = self.threshold()
            contours, hierarchy = cv2.findContours(threshold_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            len_contours = len(contours)

            if self.show_process:
                i += 1
                print(i, "len_contours =", len_contours, "threshold_level =", self.threshold_level)

            if len_contours > number_of_stars:
                break

            len_contours_good = len_contours
            threshold_level_good = self.threshold_level_stars
            self.threshold_level_stars += 1

        if len_contours_good >= number_of_stars:
            self.threshold_level_stars = threshold_level_good

        self.contours = contours

        return threshold_img

    # Get the moments   FORCE
    def moments(self, img=None, number_of_stars=None):
        if self.show_process:
            print("moments")

        if number_of_stars is None:
            number_of_stars = self.number_of_stars

        if not hasattr(self, "contours"):
            self.threshold_plus_findContours()

        if len(self.contours) < number_of_stars:
            self.threshold_plus_findContours()

        sorted_contours = sorted(self.contours, key=cv2.contourArea, reverse=True)
        max_contourArea = cv2.contourArea(sorted_contours[number_of_stars - 1])

        starsCenters_x = []
        starsCenters_y = []
        for contour in self.contours:
            if cv2.contourArea(contour) >= max_contourArea:
                mom = cv2.moments(contour)
                starsCenters_x.append(mom["m10"] / (mom["m00"] + 1e-5))
                starsCenters_y.append(mom["m01"] / (mom["m00"] + 1e-5))

        self.starsCenters_x = starsCenters_x
        self.starsCenters_y = starsCenters_y

        if self.video or self.show_process:
            if img is None:
                img = self.im_gray.copy()
                pass

            for i in range(number_of_stars):
                cv2.circle(
                    img,
                    (int(self.starsCenters_x[i]), int(self.starsCenters_y[i])),
                    radius=2,
                    color=(0, 0, 255),
                    thickness=2,
                )  # нар
                cv2.putText(
                    img,
                    str(i),
                    (int(self.starsCenters_x[i]), int(self.starsCenters_y[i])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (25, 25, 25),
                    1,
                )

            if self.show_process:
                self.def_imShow(img)

        return img

    # центральная линия FORCE
    def center_line(self, img=None, y_shift=None):
        if self.show_process:
            print("center_line")

        # сдвиг засветки на паттерне
        if not hasattr(self, "y_shift"):
            if y_shift is not None:
                self.y_shift = y_shift
            else:
                self.y_shift = (self.im_w + self.im_h) / 500  # 5

        if not hasattr(self, "starsCenters_x"):
            self.moments()

        # шаг увеличения Y на пикселе
        self.delta_x = self.starsCenters_x[self.star2] - self.starsCenters_x[self.star1]
        self.delta_y = self.starsCenters_y[self.star2] - self.starsCenters_y[self.star1]
        self.y_1step = self.delta_y / self.delta_x

        # начало линии с левого края
        self.y0 = self.starsCenters_y[self.star1] - self.starsCenters_x[self.star1] * self.y_1step
        self.y0 += self.y_shift

        # начало линии с верху
        self.x0 = self.starsCenters_x[self.star1] - self.starsCenters_y[self.star1] / self.y_1step
        self.x0 -= self.y_shift / self.y_1step

        if self.video or self.show_process:
            if img is None:
                img = self.im_gray.copy()
            cv2.line(
                img,
                (0, int(self.y0)),
                (self.im_w, int(self.y0 + self.im_w * self.y_1step)),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            # cv2.circle (img, (int( x[1] ), int( y[1])), 5, (0, 0, 255), 2) # нар
        if self.show_process:
            print("y_shift=", self.y_shift, "y_1step=", self.y_1step, "y0=", self.y0)
            self.def_imShow(img)

        return img

    def get_kikuchi_band(self, img=None):
        if self.show_process:
            print("get_kikuchi_band")

        if not hasattr(self, "y_band"):
            if not hasattr(self, "delta_x"):
                self.center_line()

            KikuchiBands.y_band = int(self.band_w * math.sqrt(self.delta_y**2 / self.delta_x**2 + 1.0))
            pass

        if self.video or self.show_process:
            if img is None:
                img = self.im_gray.copy()

            # ширина полосы
            y_band = self.y_band

            cv2.line(
                img,
                (0, int(self.y0 + y_band / 2)),
                (self.im_w, int(self.y0 + y_band / 2 + self.im_w * self.y_1step)),
                (255, 255, 255),
                2,
                lineType=6,
            )
            cv2.line(
                img,
                (0, int(self.y0 - y_band / 2)),
                (self.im_w, int(self.y0 - y_band / 2 + self.im_w * self.y_1step)),
                (255, 255, 255),
                2,
                cv2.LINE_8,
            )

        if self.show_process:
            print("self.star2=", self.star2, "self.star1=", self.star1)
            print("self.starsCenters_x=", self.starsCenters_x, "self.starsCenters_y=", self.starsCenters_y)
            self.def_imShow(img)

        return img

    # Rotation
    def get_rotation_matrix(self):
        if self.show_process:
            print("get_rotation_matrix")

        if not hasattr(self, "param_filter_stars"):
            self.filter_stars = "GaussianBlur"
            self.param_filter_stars, self.threshold_level_stars = self.choice_stars_filter_param(filt=self.filter_stars)
            self.im_gray_ff = self.first_filter(filt=self.filter_stars, param1=self.param_filter_stars)

        # начало линии с левого края
        if not hasattr(self, "y0"):
            img = self.center_line()

        self.bsbc_rotation_center = (self.im_w / 2, self.y0 + self.y_1step * self.im_w / 2)
        angle = 180.0 / np.pi * np.arctan(self.y_1step)
        self.bsbc_rotation_matrix = cv2.getRotationMatrix2D(self.bsbc_rotation_center, angle, 1.0)

        if self.show_process:
            img = self.center_line()
            cv2.circle(
                img,
                (round(self.bsbc_rotation_center[0]), round(self.bsbc_rotation_center[1])),
                radius=2,
                color=(0, 0, 255),
                thickness=2,
            )  # нар
            cv2.putText(
                img,
                "bsbc_rotation_center",
                (round(self.bsbc_rotation_center[0]), round(self.bsbc_rotation_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (25, 25, 25),
                1,
            )
            self.def_imShow(img)

        return self.bsbc_rotation_matrix

    def rotation_img(self):
        if self.show_process:
            print("rotation_img")

        img = self.im_gray_f.copy()

        if not hasattr(self, "bsbc_rotation_matrix"):
            self.bsbc_rotation_matrix = self.get_rotation_matrix()

        img_rotated = cv2.warpAffine(img, self.bsbc_rotation_matrix, (self.im_w, self.im_h))
        img_rotated = self.float255(img_rotated)

        if self.show_process:
            self.def_imShow(img_rotated)

        return img_rotated

    def quality_parameter(self, img):
        if self.show_process:
            print("quality_parameter")

        # ширина полосы
        band_w = self.band_w
        a_p = self.averaging_position

        if not hasattr(self, "bsbc_rotation_center"):
            self.get_rotation_matrix()

        nose = img[
            int(self.bsbc_rotation_center[1] - band_w / 2) : int(self.bsbc_rotation_center[1] + band_w / 2),
            a_p - self.averaging // 2 : a_p + self.averaging // 2,
        ]

        #  сумма элементов в строках
        values_y = np.sum(nose, axis=1)
        values_y /= self.averaging

        xN = np.arange(0, band_w, 1)

        background = self.im_gray_f.mean()

        self.bandContrast = values_y.max() - background

        y_max = int(values_y[0 : band_w // 2].max())
        y_min = int(values_y[0 : band_w // 2].min())

        img_short = np.zeros((y_max - y_min + 1, band_w // 2), dtype="uint8")

        for i in range(band_w // 2):
            img_short[int(values_y[i] - y_min), i] = 255

        if self.show_process:
            print("averaging=", self.averaging)

            # показываем нос
            self.def_imShow(nose)

            # красим нос белым
            nose[0:band_w, 0 : self.averaging] = 255

            cv2.line(
                img,
                (0, int(self.bsbc_rotation_center[1] + band_w / 2)),
                (self.im_w, int(self.bsbc_rotation_center[1] + band_w / 2)),
                (255, 255, 255),
                2,
                lineType=6,
            )
            cv2.line(
                img,
                (0, int(self.bsbc_rotation_center[1] - band_w / 2)),
                (self.im_w, int(self.bsbc_rotation_center[1] - band_w / 2)),
                (255, 255, 255),
                2,
                cv2.LINE_8,
            )

            title = ""
            # pic_box = plt.figure() #
            pic_box = plt.figure(figsize=(20, 5))

            pic_box.add_subplot(131).set_title(title + " shape=" + str(img.shape))
            plt.imshow(img, cmap="gray")

            pic_box.add_subplot(132)
            plt.plot(xN, values_y, xN, background + 0 * xN)

            pic_box.add_subplot(133).set_title(title + " shape=" + str(img_short.shape))
            plt.imshow(img_short, cmap="gray")

            fileName = "band_rotated_n" + str(self.averaging) + ".png"
            plt.savefig(fileName, dpi=200)
            plt.show()

            print(
                "Band Contrast (BC)",
                self.bandContrast,
                "=",
                y_max,
                "-",
                background,
                "averaging=",
                self.averaging,
                "fileName=",
                self.file_name,
            )

        return img_short, img

    def frange(self, start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    # def houghLines(self, img_short, number_of_lines=1, min_theta=np.pi/2.+1):
    def houghLines(self, img_short, number_of_lines=1, min_theta=np.pi / 2.0):
        if self.show_process:
            print("houghLines")

        lines_good = None
        threshold_good = 0
        lenLines_good = 100000000

        rho_start = 1.0
        threshold_start = 1

        dst = self.floaf_to_int255(img_short)
        rho = rho_start
        theta = np.pi / 1800.0

        for threshold in range(threshold_start, 1000):
            lines = cv2.HoughLines(dst, rho, theta, threshold, None, 0, 0, min_theta, np.pi - 0.1)
            # lines: A vector that will store the parameters (r,θ) of the detected lines
            # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
            # rho : The resolution of the parameter r in pixels. We use 1 pixel.
            # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)

            # threshold: The minimum number of intersections to "*detect*" a line
            # srn and stn: Default parameters to zero. Check OpenCV reference for more info.
            # lines
            # srn	For the multi-scale Hough transform, it is a divisor for the distance resolution rho. The coarse
            # accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn.
            # If both srn=0 and stn=0, the classical Hough transform is used.
            # Otherwise, both these parameters should be positive.
            # stn	For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
            # min_theta	For standard and multi-scale Hough transform, minimum angle to check for lines.
            # Must fall between 0 and max_theta.
            # max_theta	For standard and multi-scale Hough transform, an upper bound for the angle.
            # Must fall between min_theta and CV_PI.
            # The actual maximum angle in the accumulator may be slightly less than max_theta,
            # depending on the parameters min_theta and theta.

            if lines is None:
                break

            lines_good = lines
            lenLines = len(lines)

            if lenLines > lenLines_good:
                # print("can not be 1", "self.fileName=", self.fileName,
                # "lenLines=", lenLines, "lenLines_good=", lenLines_good)
                pass

            if lenLines_good < number_of_lines:
                break

            lenLines_good = lenLines
            threshold_good = threshold

        if lenLines_good > 1:
            threshold = threshold_good + 1

            for rho in self.frange(rho_start, 2.1, 0.05):
                lines = cv2.HoughLines(dst, rho, theta, threshold, None, 0, 0, min_theta, np.pi - 0.1)

                if lines is not None:
                    lenLines = len(lines)

                    if lenLines > lenLines_good:
                        # print("mir", "self.fileName=", self.fileName,
                        # "lenLines=", lenLines, "lenLines_good=", lenLines_good)
                        break

                    lenLines_good = lenLines
                    lines_good = lines
                    break

        # bandSlope = 0

        if lines_good is not None:
            theta = lines_good[0][0][1]
            self.bandSlope = -1.0 / np.tan(theta)

        if self.show_process:
            img_hough = img_short.copy()
            for i in range(0, len(lines_good)):
                rho = lines_good[i][0][0]
                theta = lines_good[i][0][1]

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

                cv2.line(img_hough, pt1, pt2, (200, 200, 200), 1, cv2.LINE_AA)

            print("bandSlope (BS)", self.bandSlope, "theta =", theta, "fileName=", self.file_name)
            print("min_theta = ", min_theta, "max_theta =", np.pi - 0.1, "theta =", theta)

            self.def_imShow(img_short, img_hough)

        self.theta = theta  # ?

    def get_bs_bc(self):
        if self.show_process:
            print("get_bs_bc")

        img_rotated = self.rotation_img()
        img_short, _ = self.quality_parameter(img_rotated)
        self.houghLines(img_short)

        return self.bandSlope, self.bandContrast
