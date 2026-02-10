import os
import tempfile
import io
import warnings
from typing import List
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from .page_model import TextBlock

# Подавляем предупреждения
warnings.filterwarnings('ignore', category=UserWarning)


class OCRFallback:

    def __init__(self, tesseract_path: str = None):
        # Используем Tesseract OCR - лучше работает с русским языком
        # Указываем путь к Tesseract, если он не в PATH
        if tesseract_path:
            # Если передан путь к папке, добавляем tesseract.exe
            if os.path.isdir(tesseract_path):
                tesseract_exe = os.path.join(tesseract_path, 'tesseract.exe')
                if os.path.exists(tesseract_exe):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
                else:
                    # Пробуем найти в подпапках
                    for subdir in ['bin', '']:
                        test_path = os.path.join(tesseract_path, subdir, 'tesseract.exe')
                        if os.path.exists(test_path):
                            pytesseract.pytesseract.tesseract_cmd = test_path
                            break
            elif os.path.isfile(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Пробуем стандартные пути установки
            default_paths = [
                r'D:\Go-prog\prog2\tesseract\tesseract.exe',
                r'D:\Go-prog\prog2\tesseract\bin\tesseract.exe',
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for path in default_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Проверяем наличие Tesseract
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                "Tesseract OCR не найден. Укажите путь к tesseract.exe:\n"
                "OCRFallback(tesseract_path='D:\\Go-prog\\prog2\\tesseract\\tesseract.exe')\n"
                "Или добавьте Tesseract в PATH"
            ) from e

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Предобработка изображения для улучшения качества OCR"""
        # Конвертируем в RGB если нужно
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Конвертируем в grayscale для лучшего распознавания
        img_gray = img.convert('L')
        
        # Увеличиваем разрешение если изображение маленькое (минимум 300 DPI)
        width, height = img_gray.size
        min_dpi = 300
        current_dpi = 72  # Стандартный DPI для PDF
        
        if width < 2000 or height < 2000:
            scale = max(2000 / width, 2000 / height)
            new_size = (int(width * scale), int(height * scale))
            img_gray = img_gray.resize(new_size, Image.Resampling.LANCZOS)
        
        # Увеличиваем контрастность
        enhancer = ImageEnhance.Contrast(img_gray)
        img_gray = enhancer.enhance(1.3)
        
        # Увеличиваем резкость
        enhancer = ImageEnhance.Sharpness(img_gray)
        img_gray = enhancer.enhance(1.5)
        
        # Применяем легкое размытие для сглаживания шумов
        img_gray = img_gray.filter(ImageFilter.MedianFilter(size=3))
        
        # Бинаризация (опционально, но часто помогает)
        # Конвертируем в numpy для бинаризации
        img_array = np.array(img_gray)
        # Адаптивная бинаризация
        threshold = np.mean(img_array)
        img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        img_gray = Image.fromarray(img_array)
        
        return img_gray

    def extract(self, image_bytes: bytes) -> List[TextBlock]:
        # Открываем изображение
        img = Image.open(io.BytesIO(image_bytes))
        
        # Предобрабатываем изображение
        processed_img = self._preprocess_image(img)
        
        # Используем Tesseract с русским и английским языками
        # PSM 6 = единый блок текста (лучше для документов)
        # OEM 3 = используем LSTM OCR engine (лучше для русского)
        custom_config = r'--oem 3 --psm 6 -l rus+eng'
        
        # Получаем данные с координатами и уверенностью
        data = pytesseract.image_to_data(
            processed_img,
            config=custom_config,
            output_type=pytesseract.Output.DICT,
            lang='rus+eng'
        )

        blocks = []
        n_boxes = len(data['text'])
        
        # Масштабируем координаты обратно, если изображение было увеличено
        scale_x = img.width / processed_img.width if processed_img.size != img.size else 1.0
        scale_y = img.height / processed_img.height if processed_img.size != img.size else 1.0
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
            
            # Пропускаем пустые строки и результаты с низкой уверенностью
            if not text or conf < 30:
                continue
            
            # Получаем координаты
            x = int(data['left'][i] * scale_x)
            y = int(data['top'][i] * scale_y)
            w = int(data['width'][i] * scale_x)
            h = int(data['height'][i] * scale_y)
            
            # Пропускаем слишком маленькие блоки (вероятно шум)
            if w < 5 or h < 5:
                continue

            blocks.append(
                TextBlock(
                    text=text,
                    bbox=(x, y, x + w, y + h),
                    font_size=h  # Используем высоту как размер шрифта
                )
            )
        
        return blocks
