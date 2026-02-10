from dataclasses import dataclass, field
from typing import List, Tuple, Optional


BBox = Tuple[float, float, float, float]


@dataclass
class TextBlock:
    text: str
    bbox: BBox
    font_size: float


@dataclass
class TableBlock:
    name: str
    bbox: BBox
    rows: List[List[str]]


@dataclass
class Line:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    width: float


@dataclass
class Stamp:
    bbox: BBox
    fields: dict


@dataclass
class PageModel:
    page_num: int
    texts: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    stamps: List[Stamp] = field(default_factory=list)
    is_scanned: bool = False

    def to_dict(self) -> dict:
        return {
            "page": self.page_num,
            "is_scanned": self.is_scanned,
            "texts": [t.__dict__ for t in self.texts],
            "tables": [t.__dict__ for t in self.tables],
            "lines": [l.__dict__ for l in self.lines],
            "stamps": [s.__dict__ for s in self.stamps]
        }
