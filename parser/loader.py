import fitz


class PdfLoader:
    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)

    def __len__(self):
        return len(self.doc)

    def pages(self):
        for page in self.doc:
            yield page

    def render_page(self, page, dpi=300):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
