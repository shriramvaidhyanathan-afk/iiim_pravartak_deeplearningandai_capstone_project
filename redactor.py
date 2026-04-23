from abc import ABC, abstractmethod
from typing import Any
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

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
        conf = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}],
        }
        provider = NlpEngineProvider(nlp_configuration=conf)
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        self.anonymizer = AnonymizerEngine()

        # Define specific behaviors for common PII
        self.operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"}),
            "ORG": OperatorConfig("replace", {"new_value": "<COMPANY>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONENUMBER>"}),
        }

    def redact(self, text: str) -> str:
        if not text.strip():
            return text

        results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=["PERSON", "ORG", "EMAIL_ADDRESS", "LOCATION", "PHONE_NUMBER"]
        )

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

