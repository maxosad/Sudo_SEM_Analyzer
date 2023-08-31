from typing import List, Tuple

from pydantic import BaseModel


class StackingResponseSchema(BaseModel):
    value1: str
    value2: bytes


class StackingResponseSchema1(BaseModel):
    file_id: int
    coordinates: List[Tuple[int, List[List[float]]]]


class KikuchiResponseSchema1(BaseModel):
    file_id: int


class ImageDownloadedArr(BaseModel):
    list_id: List[int]
