"""
Vision-модель для вопросов по изображениям.
Используется Qwen2-VL — работает без flash_attn (подходит для Mac).
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Модель 2B — меньше памяти, подходит для MacBook. Вариант 7B: "Qwen/Qwen2-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def load_model():
    """Загрузка без flash_attn (по умолчанию)."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Не указываем attn_implementation="flash_attention_2" — тогда flash_attn не нужен
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return model, processor


def ask_images(model, processor, image_paths, question, max_new_tokens=256):
    """Вопрос по одному или нескольким изображениям. image_paths — список путей к файлам."""
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "path": p} for p in image_paths],
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0] if output_text else ""


if __name__ == "__main__":
    print("Загрузка Qwen2-VL (без flash_attn)...")
    model, processor = load_model()

    images = [
        "./examples/docowl2_page0.png",
        "./examples/docowl2_page1.png",
        "./examples/docowl2_page2.png",
    ]
    # Проверяем существование файлов; если папки examples нет — используем один тестовый путь
    import os
    existing = [p for p in images if os.path.isfile(p)]
    if not existing:
        print("Файлы в ./examples/ не найдены. Укажите свои пути в списке images.")
        exit(1)
    images = existing

    q1 = "What is this document about? Provide a short description."
    print("Вопрос:", q1)
    answer = ask_images(model, processor, images[:1], q1)
    print("Ответ:", answer)

    q2 = "What is on the first page? Briefly."
    print("\nВопрос:", q2)
    answer2 = ask_images(model, processor, images[:1], q2)
    print("Ответ:", answer2)
