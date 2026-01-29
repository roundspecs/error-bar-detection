from src.common.models import DatasetLine, DatasetPoint, ImageAnnotation
from typing import List

def remove_points_with_long_barheight(dataset: List[ImageAnnotation]) -> List[ImageAnnotation]:
    """Removes points where the distance to top or bottom bar is greater than 430 pixels."""
    MAX_BARHEIGHT = 430.0
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

def remove_points_near_phantoms(dataset: List[ImageAnnotation]) -> List[ImageAnnotation]:
    """Removes points that are too close to phantom points."""
    MIN_DISTANCE_TO_PHANTOM = 10
    total_removed = 0

    for image in dataset:
        phantom_points = {(p.x, p.y) for line in image.lines for p in line.points if p.label in ['xmin', 'xmax', 'ymin', 'ymax']}
        for line in image.lines:
            for point in line.points:
              if point.label in ['xmin', 'xmax', 'ymin', 'ymax']:
                  continue
              distance = min(
                  ((point.x - px) ** 2 + (point.y - py) ** 2) ** 0.5
                  for (px, py) in phantom_points
              ) if phantom_points else float('inf')
              if distance < MIN_DISTANCE_TO_PHANTOM:
                  line.points.remove(point)
                  total_removed += 1
    print(f"Total points removed near phantoms: {total_removed}")
    return dataset

def remove_points_with_error_bars_beyond_image(dataset: List[ImageAnnotation]) -> List[ImageAnnotation]:
    """Removes points whose error bars extend beyond phantom points."""
    TOLERANCE = 15
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
                if upper_bar < min_phantom_y - TOLERANCE or lower_bar > max_phantom_y + TOLERANCE:
                    total_removed += 1
                    continue
                filtered_points.append(point)
            line.points = filtered_points
    print("Total points removed due to error bars beyond image:", total_removed)
    return dataset