from typing import List
from .page_model import TextBlock


class TextExtractor:

    def extract(self, page) -> List[TextBlock]:
        blocks = []
        data = page.get_text("dict")["blocks"]

        for block in data:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    blocks.append(
                        TextBlock(
                            text=text,
                            bbox=tuple(span["bbox"]),
                            font_size=span["size"]
                        )
                    )

        return blocks
