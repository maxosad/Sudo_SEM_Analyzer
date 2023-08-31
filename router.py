import os
from enum import Enum
from typing import Annotated, Any, List

import cv2

# import cv
from fastapi import APIRouter, Depends

from sem_analyzer_server.editor.stacking.KikuchiLines import KikuchiLines
from sem_analyzer_server.editor.stacking.schemas import (
    ImageDownloadedArr,
    KikuchiResponseSchema1,
    StackingResponseSchema,
    StackingResponseSchema1,
)
from sem_analyzer_server.editor.stacking.service import StackingService
from sem_analyzer_server.files.service import FilesService

from ...files.types import BinaryId
from .Stacking import Stacking

router = APIRouter(prefix="/stacking")


@router.get("/", response_model=StackingResponseSchema)
async def example_stacking_router(stacking_service: Annotated[StackingService, Depends()]) -> Any:
    return stacking_service.example_method("example_value")


@router.get("/input", response_model=ImageDownloadedArr)
async def input(
    files_service: Annotated[FilesService, Depends()], stacking_service: Annotated[StackingService, Depends()]
) -> Any:
    arr = []
    directory = "c:\\Users\\Maxim\\Studing\\MyDiplom\\data\\Bad_dataset\\"
    tiff = ".tiff1"

    for j in range(1, 100 + 1):
        if tiff == ".tiff":
            fileName = "rim_" + f"{j:03}.tiff"
        else:
            fileName = str(j) + ".bmp"

        imagename = directory + fileName
        if not os.path.exists(imagename):
            continue
        im = cv2.imread(imagename)
        im_coded = cv2.imencode(".bmp", im)[1]
        file = await files_service.create_file(name="ki_lines.bmp", content=im_coded.tobytes())
        arr.append(file.id)
    return ImageDownloadedArr(list_id=arr)


@router.get("/kikuchi", response_model=KikuchiResponseSchema1)
async def kikuchi_router(
    file_id: BinaryId,
    files_service: Annotated[FilesService, Depends()],
    stacking_service: Annotated[StackingService, Depends()],
    alg: int = 3,
) -> KikuchiResponseSchema1:
    # file_id = 1
    img_path = await files_service.get_file_by_id(file_id)
    img = cv2.imread(img_path.binary.path)
    print(img.shape, img_path.binary.path)

    KikuchiLines.dataset = ""

    KikuchiLines.number_of_stars = 3
    KikuchiLines.star1 = 1
    KikuchiLines.star2 = 2

    if img is not None:
        if len(img.shape) > 2:
            imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            imGray = img

        pattern = KikuchiLines(im_gray=imGray)
    else:
        pass
        # pattern = Kikuchi_lines(fileName=fileName)

    # alg = 3  # 1,2,3
    im_color, args1, args2 = pattern.get_kikuchi_lines(alg)

    print(im_color, args1, args2)
    im_coded = cv2.imencode(".bmp", im_color)[1]
    file = await files_service.create_file(name="ki_lines.bmp", content=im_coded.tobytes())

    return KikuchiResponseSchema1(file_id=file.id)


class AlgName(str, Enum):
    keypoints = "keypoints"
    stars_center = "stars_center"
    transformECC = "transformECC"
    correlation = "correlation"
    full_scan = "dataset_full_scan"


@router.post("/stack", response_model=StackingResponseSchema1)
async def stack_router(
    list_id: List[int],
    files_service: Annotated[FilesService, Depends()],
    stacking_service: Annotated[StackingService, Depends()],
    algorithm: AlgName = AlgName.full_scan,
) -> StackingResponseSchema1:
    # algorithm = "keypoints"  # 4
    # algorithm = "stars_center"  # 4.1
    # algorithm = "transformECC"  # 5
    # algorithm = "correlation"  # 6
    # algorithm = "dataset_full_scan"  # 6.1

    # fileName = "rim_001.tiff"
    Stacking.dataset = "EBSD_pentlandite"
    Stacking.filter = "medianBlur"
    Stacking.param_filter = 5

    Stacking.y_shift = 5

    Stacking.number_of_stars = 3
    Stacking.star1 = 1
    Stacking.star2 = 2

    dataset_list = []

    for id in list_id:
        img_path = await files_service.get_file_by_id(id)
        img = cv2.imread(img_path.binary.path)

        if len(img.shape) > 2:
            im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = img

        patternJ = Stacking(im_gray=im_gray)
        patternJ.id = id
        dataset_list.append(patternJ)

    pattern1 = dataset_list[0]
    Stacking.im_w = pattern1.im_gray.shape[1]  # ширина
    Stacking.im_h = pattern1.im_gray.shape[0]  # высота
    Stacking.band_w = round((pattern1.im_w + pattern1.im_h) / 30)  # (1024+1344)/30

    if pattern1.im_w < 700:
        # fileName = "11.bmp"
        Stacking.dataset = "Bad_dataset"
        Stacking.filter = "GaussianBlur"
        Stacking.param_filter = 95

    # 4
    if algorithm == "keypoints":
        detector_name = "ORB"
        detector_name = "SIFT"
        Stacking.detector_name = detector_name

        descriptor_matcher_name = "DESCRIPTOR_MATCHER_BRUTEFORCE"
        descriptor_matcher_name = "DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING"
        # descriptor_matcher_name = 'DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT'
        # descriptor_matcher_name = 'DESCRIPTOR_MATCHER_BRUTEFORCE_L1'
        # descriptor_matcher_name = 'DESCRIPTOR_MATCHER_BRUTEFORCE_SL2'
        # descriptor_matcher_name = 'DESCRIPTOR_MATCHER_FLANNBASED'
        Stacking.descriptor_matcher_name = descriptor_matcher_name

        method_name = "RANSAC"  # RANSAC-based robust method
        # method_name = 'LMEDS' #  Least-Median robust method
        # method_name = 'RHO' # PROSAC-based robust method
        Stacking.method_name = method_name
        Stacking.apertura = 20

        iTrain = 0
        pattern_train = dataset_list[iTrain]
        pattern_train.dataset_warpPerspective(dataset_list)

    # 4.1
    if algorithm == "stars_center":
        Stacking.apertura = 10

        iTrain = 0
        pattern_train = dataset_list[iTrain]
        pattern_train.dataset_stars_center(dataset_list)

    # 5
    if algorithm == "transformECC":
        # Define the motion model
        Stacking.modeName = "MOTION_EUCLIDEAN"
        Stacking.warp_mode = cv2.MOTION_EUCLIDEAN
        Stacking.modeName = "MOTION_AFFINE"
        Stacking.warp_mode = cv2.MOTION_AFFINE
        Stacking.modeName = "MOTION_HOMOGRAPHY"
        Stacking.warp_mode = cv2.MOTION_HOMOGRAPHY
        # Stacking.modeName = 'MOTION_TRANSLATION'; Stacking.warp_mode = cv2.MOTION_TRANSLATION

        iTrain = 0
        pattern_train = dataset_list[iTrain]
        pattern_train.dataset_transformECC(dataset_list)

    # 6
    if algorithm == "correlation":
        Stacking.frame = 100
        Stacking.apertura = 6

        iTrain = 0
        pattern_train = dataset_list[iTrain]
        pattern_train.dataset_correlation(dataset_list)

    # 6.1
    if algorithm == "dataset_full_scan":
        Stacking.frame = 100
        Stacking.apertura = 6

        iTrain = 0
        pattern_train = dataset_list[iTrain]
        pattern_train.dataset_full_scan(dataset_list)

    # stack_img, _ = pattern_train.stack(dataset_list)
    stack_img, coord = pattern_train.stack(dataset_list)
    # stack_img = pattern_train.stack(dataset_list)
    # print(stack_img)
    # im_color = cv2.imencode(".bmp", im_color)[1]
    # await files_service.create_file(name="ki_lines.bmp", content=im_color.tobytes())

    im_coded = cv2.imencode(".bmp", stack_img)[1]
    stack_file = await files_service.create_file(name="stack.bmp", content=im_coded.tobytes())
    # print('------------------------------------------')
    # print("coord ",type(coord[0][1]), coord)
    # coord = [
    #     (1,[[1.0,2.0],[3.0,4.0]]),
    #     (2,[[5.0,6.0],[7.0,8.0]])
    # ]
    #
    # print("coord2 ",type(coord[0][1]), coord)
    return StackingResponseSchema1(file_id=stack_file.id, coordinates=coord)

    # return StackingResponseSchema1(value1=1)
