from src.common.models import DatasetLine, DatasetPoint, ImageAnnotation
from config import MAX_BARHEIGHT, MINIMUM_PHANTOM_DISTANCE
from typing import List

def remove_points_with_long_barheight(dataset: List[ImageAnnotation]) -> List[ImageAnnotation]:
    """Removes points where the distance to top or bottom bar is greater than 430 pixels."""
    total_removed = 0

    for image in dataset:
        for line in image.lines:
            filtered_points = [
                point for point in line.points
                if point.topBarPixelDistance + point.bottomBarPixelDistance <= MAX_BARHEIGHT
            ]
            total_removed += len(line.points) - len(filtered_points)
            line.points = filtered_points

    print("Total points removed due to long barheight:", total_removed)
    return dataset

def remove_points_with_error_bars_beyond_image(dataset: List[ImageAnnotation]) -> List[ImageAnnotation]:
    """Removes points whose error bars extend beyond phantom points."""
    total_removed = 0

    for image in dataset:
        phantom_y = [p.y for line in image.lines for p in line.points if p.label in ['xmin', 'xmax', 'ymin', 'ymax']]
        if not phantom_y:
            continue
        min_phantom_y = min(phantom_y)
        max_phantom_y = max(phantom_y)
        for line in image.lines:
            filtered_points = []
            for point in line.points:
                upper_bar = point.y - point.topBarPixelDistance
                lower_bar = point.y + point.bottomBarPixelDistance
                if upper_bar < min_phantom_y - MINIMUM_PHANTOM_DISTANCE or lower_bar > max_phantom_y + MINIMUM_PHANTOM_DISTANCE:
                    total_removed += 1
                    continue
                filtered_points.append(point)
            line.points = filtered_points
    print("Total points removed due to error bars beyond image:", total_removed)
    return dataset