from abc import ABC, abstractmethod
from typing import Any, Union
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


class BaseRedactor(ABC):
    """
    Abstract Base Class for all redaction services.
    Ensures a consistent interface across text and visual data.
    """

    @abstractmethod
    def redact(self, data: Any) -> Any:
        """
        Main method to perform redaction.
        :param data: The raw input (str for text, bytes for images)
        :return: The sanitized output
        """
        pass


class TextRedactor(BaseRedactor):
    def __init__(self):
        # Initialize engines once to save memory on M1 Pro
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Define specific behaviors for common PII
        self.operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<REDACTED_NAME>"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 10, "from_end": True}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<REDACTED_EMAIL>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<REDACTED_LOC>"}),
        }

    def redact(self, text: str) -> str:
        if not text.strip():
            return text

        results = self.analyzer.analyze(text=text, language='en')
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self.operators
        )
        return anonymized.text


class ImageRedactor(BaseRedactor):
    def __init__(self, text_redactor: TextRedactor):
        # We inject the TextRedactor to reuse the Presidio logic
        self.text_redactor = text_redactor

    def redact(self, image_bytes: bytes) -> bytes:
        # 1. Convert bytes to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Get OCR data including bounding boxes
        # Output.DICT gives us 'text', 'left', 'top', 'width', 'height'
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)

        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            word = ocr_data['text'][i].strip()
            if not word:
                continue

            # 3. Check if this specific word is PII
            # We check if the redactor modifies the word
            if self._is_pii(word):
                (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])

                # 4. Draw black box over the PII
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

        # 5. Convert back to bytes
        _, encoded_img = cv2.imencode('.png', img)
        return encoded_img.tobytes()

    def _is_pii(self, text: str) -> bool:
        """Helper to check if a word contains PII using the TextRedactor."""
        redacted = self.text_redactor.redact(text)
        return "<REDACTED" in redacted or "*" in redacted

