from io import BytesIO

import jiwer
import pytesseract

from PIL import Image
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def process_image_with_paddle(numpy_image, lang="uk", return_only_text=False):
    """Get as input image in bytes, pass into PaddleOCR and return output in bytes or text"""
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    result = ocr.ocr(numpy_image, cls=True)

    txts = [] if not result[0] else [line[1][0] for line in result[0]]
    if return_only_text:
        return txts
    else:
        if not result[0]:
            boxes, scores = [], []
        else:
            boxes = [line[0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]

    image = Image.fromarray(numpy_image).convert('RGB')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='fonts/cyrillic.ttf')
    im_show = Image.fromarray(im_show)
    im_show.show()
    bio = BytesIO()
    bio.name = 'image.jpeg'
    im_show.save(bio, 'JPEG')
    bio.seek(0)
    return bio


def process_image_with_tesseract(image):
    result = pytesseract.image_to_string(image, lang="ukr+eng")
    return result.split()


def get_accuracy_metrics(ground_truth: str, recognized: str, visualise_alignment=False):
    """Get word and character error rate between two strings"""
    words_output = jiwer.process_words(ground_truth, recognized)
    chars_output = jiwer.process_characters(ground_truth, recognized)

    if visualise_alignment:
        print(jiwer.visualize_alignment(words_output))
        print(jiwer.visualize_alignment(chars_output))
    return words_output.wer, chars_output.cer
