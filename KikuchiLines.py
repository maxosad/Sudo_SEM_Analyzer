import math

import cv2  # библиотека работ с компьютерным зрением
import numpy as np  # библиотека работы с матрицами
from scipy.optimize import curve_fit  # importing the required packages and libraries

from .KikuchiBands import KikuchiBands

# Script information for the file.
__author__ = "Maxim Osadchy (TG @MaxOsad)"
__version__ = "23.0419"
__date__ = "Apr 19, 2023"
__copyright__ = "Copyright (c) 2022 Maxim Osadchy"


class KikuchiLines(KikuchiBands):
    video = False
    show_process = False  # troubleshooting system
    number_of_stars = 3

    def __init__(self, file_name=None, im_gray=None):
        super().__init__(file_name, im_gray)

    def set_y_band(self, band_w=None):
        if band_w is None:
            band_w = self.band_w

        if not hasattr(self, "delta_x"):
            self.center_line()

        self.y_band = band_w * math.sqrt(self.delta_y**2 / self.delta_x**2 + 1.0)
        self.x_band = band_w * math.sqrt(self.delta_x**2 / self.delta_y**2 + 1.0)

        return self.y_band

    def points_sampling_algorithm_1(self, line_mask):
        if self.show_process:
            print("points_sampling_algorithm_1")

        param_filter = 1 + 2 * 1

        #         if hasattr(self, "filter"):
        #             if self.filter == 'GaussianBlur':
        #                 if hasattr(self, "param_filter"):
        #                     if self.param_filter == param_filter:
        #                         if hasattr(self, "im_gray_ff"):
        #                             pass
        #         else:

        self.param_filter = param_filter
        self.filter = "GaussianBlur"
        self.im_gray_ff = self.first_filter()

        self.im_gray_ff = self.floaf_to_int255(self.im_gray_ff)
        im_gray_ff_max = self.im_gray_ff.max()
        mask_sum = line_mask.sum() // 255

        blockSize = 1 + int(self.band_w * 2)

        for i in range(255):
            c = i
            thresh = cv2.adaptiveThreshold(
                self.im_gray_ff,
                maxValue=im_gray_ff_max,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=blockSize,
                C=c,
            )

            thresh_mask_sum = thresh[line_mask == 255].sum() // 255

            if thresh_mask_sum / mask_sum < 0.08:
                break

        img = cv2.bitwise_and(thresh, thresh, mask=line_mask)
        im = img // 255

        sum_1 = int(np.sum(im))

        values_y = np.array(range(sum_1))
        values_x = np.array(range(sum_1))

        i = 0
        for iy in range(self.im_h):
            if np.sum(im[iy]) < 1:
                continue

            for ix in range(self.im_w):
                if im[iy][ix] >= 1:
                    values_y[i] = iy
                    values_x[i] = ix
                    i += 1

        if self.show_process:
            y_band = self.y_band  # ширина полосы
            y0 = self.y0
            y_1step = self.y_1step
            im_w = self.im_w
            im_h = self.im_h

            im_top = min(round(y0 - y_band / 2 + im_w * y_1step), round(y0 - y_band / 2))
            im_bottom = max(round(y0 + y_band / 2 + im_w * y_1step), round(y0 + y_band / 2))

            im_top = min(round(y0 - y_band + im_w * y_1step), round(y0 - y_band))
            im_bottom = max(round(y0 + y_band + im_w * y_1step), round(y0 + y_band))

            if im_top < 0:
                im_top = 0

            if im_bottom > im_h:
                im_bottom = im_h

            self.def_imShow(thresh[im_top:im_bottom, 0:im_w])

            img = cv2.bitwise_and(thresh, thresh, mask=line_mask)
            self.def_imShow(img[im_top:im_bottom, 0:im_w])

        return values_x, values_y

    def points_sampling_algorithm_2(self, line_mask, param=1):
        if not hasattr(self, "im_gray_ff"):
            self.im_gray_ff = self.first_filter(self.im_gray_f)

        # im_gray_ff = self.im_gray_ff
        img = self.im_gray_ff
        # img = cv2.bitwise_and(im_gray_ff, im_gray_ff, mask=line_mask)

        # размеры картинки
        im_w = self.im_w
        im_h = self.im_h

        # начало линии с левого края
        y0 = self.y0

        # ширина полосы
        y_band = self.y_band

        band_w = self.band_w

        # dy/dx
        y_1step = self.y_1step

        values_y = np.array(range(im_w + im_h))
        values_x = np.array(range(im_w + im_h))

        x_mask_star_1 = range(
            int(self.starsCenters_x[self.star1] - band_w / 2), int(self.starsCenters_x[self.star1] + band_w / 2)
        )
        x_mask_star_2 = range(
            int(self.starsCenters_x[self.star2] - band_w / 2), int(self.starsCenters_x[self.star2] + band_w / 2)
        )

        # обход по всем x
        for ix in range(im_w):
            values_x[ix] = ix
            values_y[ix] = -1

            if ix in x_mask_star_1:
                continue
            if ix in x_mask_star_2:
                continue

            # верхняя линия
            if param == 1:
                # направление сканирования
                step = -1

                # точка окончания сканикования вверх по колонке с округлением
                iy_end = round(y0 - y_band / 2 + ix * y_1step)
                if iy_end < 0:
                    continue
                if iy_end >= im_h:
                    continue

                # точка начала сканикования вверх по колонке с округлением в меньшую сторону
                iy_start = math.floor(y0 + ix * y_1step) - 1
                if iy_start >= im_h:
                    continue
                if iy_start < iy_end:
                    continue

            # нижняя линия
            else:
                # направление сканирования
                step = 1

                # точка окончания сканикования вниз по колонке с округлением
                iy_end = round(y0 + y_band / 2 + ix * y_1step)
                if iy_end >= im_h:
                    continue
                if iy_end < 0:
                    continue

                # точка начала сканикования вниз по колонке с округлением в большую
                iy_start = math.ceil(y0 + ix * y_1step) + 1
                if iy_start < 0:
                    continue
                if iy_start > iy_end:
                    continue

            # начальный минимум
            imdYX_min = 255.0

            for iy in range(iy_start, iy_end + step, step):
                imdYX = img[iy][ix]

                if imdYX < imdYX_min:
                    imdYX_min = imdYX
                    values_y[ix] = iy

        # начало линии с верху
        x0 = self.x0

        # ширина полосы
        x_band = self.x_band

        y_mask_star_1 = range(
            int(self.starsCenters_y[self.star1] - band_w / 2), int(self.starsCenters_y[self.star1] + band_w / 2)
        )
        y_mask_star_2 = range(
            int(self.starsCenters_y[self.star2] - band_w / 2), int(self.starsCenters_y[self.star2] + band_w / 2)
        )

        # обход по всем y
        for iy in range(im_h):
            iy_im_w = iy + im_w
            values_y[iy_im_w] = iy
            values_x[iy_im_w] = -1

            if iy in y_mask_star_1:
                continue
            if iy in y_mask_star_2:
                continue

            # верхняя линия
            if param == 1:
                # направление сканирования
                step = 1

                # точка окончания сканикования вправо по строке с округлением
                ix_end = round(x0 + x_band / 2 + iy / y_1step)
                if ix_end >= im_w:
                    continue
                if ix_end < 0:
                    continue

                # точка начала сканикования вправо по строке с округлением в большую сторону
                ix_start = math.ceil(x0 + iy / y_1step) - 1
                if ix_start < 0:
                    continue
                if ix_start > ix_end:
                    continue

            # нижняя линия
            else:
                # направление сканирования
                step = -1

                # точка окончания сканикования вниз по колонке с округлением
                ix_end = round(x0 - x_band / 2 + iy / y_1step)
                if ix_end < 0:
                    continue
                if ix_end >= im_w:
                    continue

                # точка начала сканикования вниз по колонке с округлением в большую
                ix_start = math.ceil(x0 + iy / y_1step) + 1
                if ix_start >= im_w:
                    continue
                if ix_start < iy_end:
                    continue

            # начальный минимум
            imdYX_min = 255.0

            for ix in range(ix_start, ix_end + step, step):
                imdYX = img[iy][ix]

                if imdYX < imdYX_min:
                    imdYX_min = imdYX
                    values_x[iy_im_w] = ix

        values_y = values_y[0 < values_x[:]]
        values_x = values_x[0 < values_x[:]]
        values_x = values_x[0 < values_y[:]]
        values_y = values_y[0 < values_y[:]]

        return values_x, values_y

    def points_sampling_algorithm_3(self, line_mask, param=1):
        if not hasattr(self, "im_gray_ff"):
            self.im_gray_ff = self.first_filter(self.im_gray_f)

        if not hasattr(self, "starsCenters_x"):
            self.moments()

        # im_gray_ff = self.im_gray_ff
        img = self.im_gray_ff

        # размеры картинки
        im_w = self.im_w
        im_h = self.im_h

        # начало линии с левого края
        y0 = self.y0

        # ширина полосы
        y_band = self.y_band

        band_w = self.band_w

        # dy/dx
        y_1step = self.y_1step

        values_y = np.array(range(im_w + im_h))
        values_x = np.array(range(im_w + im_h))

        x_mask_star_1 = range(
            int(self.starsCenters_x[self.star1] - band_w / 2), int(self.starsCenters_x[self.star1] + band_w / 2)
        )
        x_mask_star_2 = range(
            int(self.starsCenters_x[self.star2] - band_w / 2), int(self.starsCenters_x[self.star2] + band_w / 2)
        )

        # обход по всем x
        for ix in range(im_w):
            values_x[ix] = ix
            values_y[ix] = -1

            if ix in x_mask_star_1:
                continue

            if ix in x_mask_star_2:
                continue

            # верхняя линия
            if param == 1:
                # направление сканирования
                step = -1

                # точка окончания сканикования вверх по колонке с округлением
                iy_end = round(y0 - y_band / 2 + ix * y_1step)
                if iy_end < 0:
                    continue
                if iy_end >= im_h:
                    continue

                # точка начала сканикования вверх по колонке с округлением в меньшую сторону
                iy_start = math.floor(y0 + ix * y_1step) - 1
                if iy_start >= im_h:
                    continue
                if iy_start < iy_end:
                    continue

            # нижняя линия
            else:
                # направление сканирования
                step = 1

                # точка окончания сканикования вниз по колонке с округлением
                iy_end = round(y0 + y_band / 2 + ix * y_1step)
                if iy_end >= im_h:
                    continue
                if iy_end < 0:
                    continue

                # точка начала сканикования вниз по колонке с округлением в большую
                iy_start = math.ceil(y0 + ix * y_1step) + 1
                if iy_start < 0:
                    continue
                if iy_start > iy_end:
                    continue

            # начальный градиент
            max_grad = 0
            values_old = img[iy_start][ix]

            for iy in range(iy_start, iy_end + step, step):
                imdYX = img[iy][ix]
                grad = values_old - imdYX

                if grad > max_grad:
                    max_grad = grad
                    values_y[ix] = iy
                values_old = imdYX

        # начало линии с верху
        x0 = self.x0

        # ширина полосы
        x_band = self.x_band

        y_mask_star_1 = range(
            int(self.starsCenters_y[self.star1] - band_w / 2), int(self.starsCenters_y[self.star1] + band_w / 2)
        )
        y_mask_star_2 = range(
            int(self.starsCenters_y[self.star2] - band_w / 2), int(self.starsCenters_y[self.star2] + band_w / 2)
        )

        # обход по всем y
        for iy in range(im_h):
            iy_im_w = iy + im_w
            values_y[iy_im_w] = iy
            values_x[iy_im_w] = -1

            if iy in y_mask_star_1:
                continue
            if iy in y_mask_star_2:
                continue

            # верхняя линия
            if param == 1:
                # направление сканирования
                step = 1

                # точка окончания сканикования вправо по строке с округлением
                ix_end = round(x0 + x_band / 2 + iy / y_1step)
                if ix_end >= im_w:
                    continue
                if ix_end < 0:
                    continue

                # точка начала сканикования вправо по строке с округлением в большую сторону
                ix_start = math.ceil(x0 + iy / y_1step) - 1
                if ix_start < 0:
                    continue
                if ix_start > ix_end:
                    continue

            # нижняя линия
            else:
                # направление сканирования
                step = -1

                # точка окончания сканикования вниз по колонке с округлением
                ix_end = round(x0 - x_band / 2 + iy / y_1step)
                if ix_end < 0:
                    continue
                if ix_end >= im_w:
                    continue

                # точка начала сканикования вниз по колонке с округлением в большую
                ix_start = math.ceil(x0 + iy / y_1step) + 1
                if ix_start >= im_w:
                    continue
                if ix_start < iy_end:
                    continue

            # начальный минимум
            max_grad = 0
            values_old = img[iy][ix_start]

            for ix in range(ix_start, ix_end + step, step):
                imdYX = img[iy][ix]
                grad = values_old - imdYX

                if grad > max_grad:
                    max_grad = grad
                    values_x[iy_im_w] = ix
                values_old = imdYX

        values_y = values_y[0 < values_x[:]]
        values_x = values_x[0 < values_x[:]]
        values_x = values_x[0 < values_y[:]]
        values_y = values_y[0 < values_y[:]]

        return values_x, values_y

    def approximation(self, values_x, values_y):
        # defining objective functions
        def mapping(values_xx, aa, bb, cc):
            return aa * values_xx**2 + bb * values_xx + cc

        # using the curve_fit() function
        args, covar = curve_fit(mapping, values_x, values_y)

        a, b, c = args[0], args[1], args[2]

        x_fit = np.array(range(self.im_w))
        y_fit = mapping(x_fit, a, b, c)  # y_fit = a * x_fit**2 + b * x_fit + c

        xx_fit = x_fit[0 <= y_fit[:]]
        yy_fit = y_fit[0 <= y_fit[:]]

        res_x_fit = xx_fit[yy_fit[:] < self.im_h]
        res_y_fit = yy_fit[yy_fit[:] < self.im_h]

        # troubleshooting show_process
        if self.show_process:
            print("Arguments: ", args)
            print("Co-Variance: ", covar)

        return res_x_fit, res_y_fit, args

    def kikuchi_lines_show(self, x_fit1, y_fit1, x_fit2, y_fit2, im_color=None):
        if im_color is None:
            img = self.im_gray.copy()
            im_color = np.dstack((img, img, img))

        kikuchi_color = (255, 0, 0)

        for ix in range(len(x_fit1)):
            im_color[math.ceil(y_fit1[ix]), x_fit1[ix]] = kikuchi_color
            im_color[math.floor(y_fit1[ix]), x_fit1[ix]] = kikuchi_color

        for ix in range(len(x_fit2)):
            im_color[math.ceil(y_fit2[ix]), x_fit2[ix]] = kikuchi_color
            im_color[math.floor(y_fit2[ix]), x_fit2[ix]] = kikuchi_color

        if self.show_process:
            self.def_imShow(im_color)

        return im_color

    def get_kikuchi_lines(self, param=1):
        im_w = self.im_gray.shape[1]  # ширина
        im_h = self.im_gray.shape[0]  # высота

        self.band_w = round((im_w + im_h) / 30)  # (1024+1344)/30

        self.im_w = im_w
        self.im_h = im_h

        self.filter_stars = "GaussianBlur"
        self.param_filter_stars, self.threshold_level_stars = self.choice_stars_filter_param(filt=self.filter_stars)
        self.im_gray_ff = self.first_filter(filt=self.filter_stars, param1=self.param_filter_stars)

        self.threshold(threshold_level=self.threshold_level_stars)
        self.findContours()
        self.moments()

        self.center_line()

        if not hasattr(self, "y_band"):
            self.set_y_band()

        y_band = self.y_band  # ширина полосы
        y0 = self.y0
        y_1step = self.y_1step

        img_zero = np.zeros((im_h, im_w), dtype="uint8")
        line_mask1 = cv2.line(
            img_zero.copy(),
            (0, math.floor(y0 - y_band / 4)),
            (im_w, math.floor(y0 - y_band / 4 + im_w * y_1step)),
            (255, 255, 255),
            self.band_w // 2,
            cv2.LINE_AA,
        )
        line_mask2 = cv2.line(
            img_zero.copy(),
            (0, math.ceil(y0 + y_band / 4)),
            (im_w, math.ceil(y0 + y_band / 4 + im_w * y_1step)),
            (255, 255, 255),
            self.band_w // 2,
            cv2.LINE_AA,
        )

        if param == 1:
            values_x1, values_y1 = self.points_sampling_algorithm_1(line_mask1)
            values_x2, values_y2 = self.points_sampling_algorithm_1(line_mask2)
        elif param == 2:
            self.param_filter = 1 + 2 * 6
            self.filter = "GaussianBlur"

            im_gray_ff = self.first_filter()
            self.im_gray_ff = im_gray_ff

            values_x1, values_y1 = self.points_sampling_algorithm_2(line_mask1, 1)
            values_x2, values_y2 = self.points_sampling_algorithm_2(line_mask2, 2)
        else:
            self.param_filter = 1 + 2 * 6
            self.filter = "GaussianBlur"

            im_gray_ff = self.first_filter()
            self.im_gray_ff = im_gray_ff

            values_x1, values_y1 = self.points_sampling_algorithm_3(line_mask1, 1)
            values_x2, values_y2 = self.points_sampling_algorithm_3(line_mask2, 2)

        x_fit1, y_fit1, args1 = self.approximation(values_x1, values_y1)
        x_fit2, y_fit2, args2 = self.approximation(values_x2, values_y2)

        im_color = self.kikuchi_lines_show(x_fit1, y_fit1, x_fit2, y_fit2)

        return im_color, args1, args2
