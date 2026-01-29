from pydantic import BaseModel
from typing import List, Optional

class DatasetPoint(BaseModel):
    x: float
    y: float
    topBarPixelDistance: float
    bottomBarPixelDistance: float
    label: Optional[str] = None

class DatasetLine(BaseModel):
    label: dict
    points: List[DatasetPoint]

DatasetFile = List[DatasetLine]

class Point2D(BaseModel):
    x: float
    y: float

class InferenceInputLine(BaseModel):
    lineName: str
    points: List[Point2D]

class InferenceInput(BaseModel):
    image_file: str
    data_points: List[InferenceInputLine]

class ErrorBarResult(BaseModel):
    data_point: Point2D
    upper_error_bar: Point2D
    lower_error_bar: Point2D