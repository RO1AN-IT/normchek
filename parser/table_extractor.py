import pdfplumber
from typing import List
from .page_model import TableBlock


class TableExtractor:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self, page_num: int) -> List[TableBlock]:
        tables = []

        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_num]
            raw_tables = page.extract_tables()

            for idx, table in enumerate(raw_tables):
                tables.append(
                    TableBlock(
                        name=f"table_{idx+1}",
                        bbox=(0, 0, 0, 0),
                        rows=table
                    )
                )

        return tables
