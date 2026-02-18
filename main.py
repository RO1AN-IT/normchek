from parser.loader import PdfLoader
from parser.text_extractor import TextExtractor
from parser.geometry_extractor import GeometryExtractor
from parser.table_extractor import TableExtractor
from parser.stamp_detector import StampDetector
from parser.ocr_fallback import OCRFallback
from parser.page_model import PageModel
import logging
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    logger.error("Файл .env не найден")
    raise FileNotFoundError("Файл .env не найден")

PATH_OCR = os.getenv("PATH_OCR")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_scanned(page):
    return len(page.get_text().strip()) < 10


def has_readable_text(texts):
    """Проверяет, содержит ли извлеченный текст читаемые символы"""
    if not texts:
        return False
    
    # Проверяем первые несколько текстовых блоков
    sample_texts = texts[:min(5, len(texts))]
    total_chars = 0
    readable_chars = 0
    
    for text_block in sample_texts:
        text = text_block.text
        total_chars += len(text)
        # Считаем читаемые символы (буквы, цифры, пробелы, пунктуация)
        readable_chars += sum(1 for c in text if c.isprintable() and ord(c) >= 32)
    
    if total_chars == 0:
        return False
    
    # Если менее 50% символов читаемые, считаем текст нечитаемым
    return (readable_chars / total_chars) >= 0.5


pdf = PdfLoader("album.pdf")

print("Инициализация экстракторов...")
text_ext = TextExtractor()
geom_ext = GeometryExtractor()
logger.info("Инициализация OCR (это может занять время)...")
# Указываем путь к Tesseract, если он не в PATH
try:
    ocr_ext = OCRFallback(tesseract_path=PATH_OCR)
except Exception as e:
    logger.error(f"{e}")
    raise
logger.info("OCR инициализирован")
stamp_det = StampDetector()
table_ext = TableExtractor("album.pdf")

pages = []
total_pages = len(pdf)

print(f"Начинаем обработку {total_pages} страниц...")

for i, page in enumerate(pdf.pages()):
    if i > 1:
        break
    print(f"Обработка страницы {i + 1}/{total_pages}...")
    scanned = is_scanned(page)

    if scanned:
        print(f"  Страница {i + 1} - отсканированная, запуск OCR...")
        img = pdf.render_page(page)
        texts = ocr_ext.extract(img)
        print(f"  OCR завершен, найдено {len(texts)} текстовых блоков")
    else:
        print(f"  Страница {i + 1} - текстовая, извлечение текста...")
        texts = text_ext.extract(page)
        
        # Проверяем, читаемый ли текст. Если нет - используем OCR
        if not has_readable_text(texts):
            print(f"  Текст содержит нечитаемые символы, переключаемся на OCR...")
            img = pdf.render_page(page)
            texts = ocr_ext.extract(img)
            scanned = True  # Помечаем как отсканированную для корректной обработки
            print(f"  OCR завершен, найдено {len(texts)} текстовых блоков")
        else:
            print(f"  Найдено {len(texts)} текстовых блоков")

    print(f"  Извлечение геометрии...")
    lines = geom_ext.extract(page)
    print(f"  Извлечение таблиц...")
    tables = table_ext.extract(i)
    print(f"  Поиск штампов...")
    stamps = stamp_det.detect(texts, lines)

    model = PageModel(
        page_num=i + 1,
        texts=texts,
        tables=tables,
        lines=lines,
        stamps=stamps,
        is_scanned=scanned
    )

    pages.append(model)
    print(f"  Страница {i + 1} обработана\n")

print("Обработка завершена!")
if pages:
    print(pages[1].to_dict())