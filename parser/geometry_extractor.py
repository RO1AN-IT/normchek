from typing import List
import math
from .page_model import Line


class GeometryExtractor:

    def __init__(self, min_line_length: float = 5.0):
        """
        Инициализация экстрактора геометрии
        
        Args:
            min_line_length: Минимальная длина линии в пикселях для фильтрации шума
        """
        self.min_line_length = min_line_length

    def _calculate_length(self, p1: tuple, p2: tuple) -> float:
        """Вычисляет длину линии между двумя точками"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    def extract(self, page) -> List[Line]:
        lines = []
        drawings = page.get_drawings()

        for d in drawings:
            width = d.get("width", 0.1)
            for item in d["items"]:
                if item[0] == "l":
                    _, p1, p2 = item
                    p1_coords = (p1.x, p1.y)
                    p2_coords = (p2.x, p2.y)
                    
                    # Вычисляем длину линии
                    length = self._calculate_length(p1_coords, p2_coords)
                    
                    # Фильтруем: пропускаем слишком короткие линии и точки
                    if length >= self.min_line_length:
                        lines.append(
                            Line(
                                p1=p1_coords,
                                p2=p2_coords,
                                width=width
                            )
                        )
        
        return lines
