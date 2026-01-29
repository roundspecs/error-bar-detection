from pathlib import Path
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

    @property
    def line_name(self) -> str:
        return self.label.get("lineName", "Unnamed Line")

class ImageAnnotation(BaseModel):
    id: str
    image_path: Path
    json_path: Path
    lines: List[DatasetLine]

    def __str__(self):
        return f"ImageAnnotation(id={self.id})"

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