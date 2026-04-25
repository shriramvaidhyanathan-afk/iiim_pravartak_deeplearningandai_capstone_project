import os
import hashlib
from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, List
import fitz
from langchain_core.documents import Document
import logging

from ai_factory import OllamaImageCaptioner
from redactor import ImageRedactor, TextRedactor
from traceabilitymanager import trace_manager


class DocumentProcessor(ABC):
    """
    Abstract base class for processing different document types.
    Ensures a consistent interface for the Chatbot.
    """
    MAX_SIZE_MB = 100

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise IOError(f"File {file_path} does not exist")

        self._file_path = file_path
        self._file_name = os.path.basename(self._file_path)
        self._file_size = os.path.getsize(self._file_path) / (1024 * 1024)

        if not self._is_valid_size():
            raise IOError(f"File {file_path} size {self._file_size} is greater than {DocumentProcessor.MAX_SIZE_MB}")

        self._text_redactor = TextRedactor()
        self._image_redactor = ImageRedactor(self._text_redactor)
        self._image_captioner = OllamaImageCaptioner()

    @property
    def file_hash(self):
        return self._get_file_hash()

    @property
    def file_path(self):
        return self._file_path

    def _get_file_hash(self) -> str:
        """
        Computes MD5 hash by streaming the file in chunks
        to maintain a low memory footprint.
        """
        hash_md5 = hashlib.md5()
        # Using 640KB chunks as requested (655360 bytes)
        chunk_size_in_bytes = 655360
        with open(self._file_path, "rb") as f:
            while chunk := f.read(chunk_size_in_bytes):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_valid_size(self) -> bool:
        """Standard size validation for all document types."""
        return self._file_size <= DocumentProcessor.MAX_SIZE_MB

    @abstractmethod
    def get_sample_text(self, max_chars: int = 1000) -> str:
        pass

    @abstractmethod
    def load(self) -> Generator:
        """Must be implemented by subclasses to handle specific file parsing."""
        pass


class PDFDocumentProcessor(DocumentProcessor):
    """
    Specifically handles PDF files using lazy loading
    to preserve M1 Pro memory.
    """

    def get_sample_text(self, max_chars: int = 1000) -> str:
        """
        Extracts up to max_chars from the beginning of the PDF
        without loading the full document into memory.
        """

        sample_accumulator = []
        current_length = 0
        with fitz.open(self._file_path) as pdf_doc:
            for page in pdf_doc:
                page_text = page.get_text()
                sample_accumulator.append(page_text)
                current_length += len(page_text)

                # Stop as soon as we exceed the required sample size
                if current_length >= max_chars:
                    break

            # Join and truncate to exactly max_chars
            full_sample = "".join(sample_accumulator)
            sample_text = full_sample[:max_chars]
            safe_text = self._text_redactor.redact(sample_text)
            trace_manager.add_original_redacted_text(sample_text, safe_text, self._file_name, 1)
            return safe_text

    def load(self) -> Generator:
        documents = []
        if not self._file_path.lower().endswith('.pdf'):
            raise ValueError("Provided file is not a PDF.")
        with fitz.open(self._file_path) as pdf_doc:
            total_pages = len(pdf_doc)
            for page_num, page in enumerate(pdf_doc):
                # 1. Extract raw text
                raw_text = page.get_text()
                actual_page_num = page_num + 1
                # 2. Redact text using Presidio logic
                logging.info(f"Redacting text on page {page_num + 1}...")
                safe_text = self._text_redactor.redact(raw_text)

                trace_manager.add_original_redacted_text(raw_text, safe_text, self._file_name, actual_page_num)

                documents.append(Document(
                    page_content=safe_text,
                    metadata={
                        "source": self._file_name,
                        "page": actual_page_num,
                        "total_pages": total_pages,
                        "type": "text_content"
                    }))

                # process the images in the page
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = pdf_doc.extract_image(xref)
                    raw_bytes = base_image["image"]

                    # --- REDACTION STEP: Images ---
                    logging.info(f"Checking Image {img_index} on Page {actual_page_num} for PII...")
                    image_bytes = self._image_redactor.redact(raw_bytes)

                    trace_manager.add_original_redacted_image(raw_bytes, image_bytes, self._file_name,
                                                              actual_page_num, img_index)

                    logging.info(f"Captioning image on page {actual_page_num}...")

                    # 1. Use Moondream to describe the image
                    caption = self._image_captioner.caption(image_bytes)

                    safe_text = self._text_redactor.redact(caption)
                    trace_manager.add_original_redacted_text(caption, safe_text, self._file_name, actual_page_num,
                                                             img_index, True)

                    documents.append(Document(
                        page_content=f"[IMAGE DESCRIPTION]: {caption}",
                        metadata={
                            "source": self._file_name,
                            "page": actual_page_num,
                            "type": "visual_element",
                            "original_index": img_index
                        }
                    ))

            yield documents

