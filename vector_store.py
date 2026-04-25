import os
import shutil
import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_processor import DocumentProcessor
from enum import Enum
from typing import Dict, Any, List, Optional
from ai_abstractions import EmbeddingClient, ImageCaptioner
from traceabilitymanager import trace_manager


class DocumentType(Enum):
    LEGAL = "Legal"
    FINANCIAL = "Financial"
    TECHNICAL = "Technical"
    GENERAL = "General"

    @property
    def chunk_config(self) -> Dict[str, Any]:
        """
        Returns the specific chunking parameters for each type.
        """
        configs = {
            DocumentType.LEGAL: {"size": 2200, "overlap": 300},
            DocumentType.FINANCIAL: {"size": 600, "overlap": 0},
            DocumentType.TECHNICAL: {"size": 1500, "overlap": 200},
            DocumentType.GENERAL: {"size": 1000, "overlap": 150},
        }
        return configs.get(self, configs[DocumentType.GENERAL])

    @classmethod
    def from_string(cls, label: str):
        """
        Maps LLM classification strings to the Enum.
        """
        lookup = {e.value.lower(): e for e in cls}
        # Clean the LLM response and match
        clean_label = label.strip().lower()
        return lookup.get(clean_label, cls.GENERAL)


class VectorStore:
    def _cleanup(self):
        if os.path.exists(self._persistent_path):
            logging.info("--- Clearing existing database ---")
            shutil.rmtree(self._persistent_path)

    def __init__(
        self,
        embeddings: EmbeddingClient,
        persistent_path: str = "./chroma_db"
    ):
        self._persistent_path = persistent_path
        self._collections = {}
        self._text_embeddings = embeddings
        self._processed_document_hashes = set()

    def reset_persistence(self):
        """Optional explicit cleanup for local development resets."""
        self._cleanup()

    def add_document(self, document_processor: DocumentProcessor, document_type: DocumentType):
        if document_processor is None:
            raise AttributeError("Document Processor cannot be a None object")

        collection_name = os.path.basename(document_processor.file_path).replace(".", "_").replace(" ", "_")

        if collection_name in self._collections.keys():
            logging.info(f"PDF name match. Collection name {collection_name} already in Vector Store")
            return

        file_hash = document_processor.file_hash
        if file_hash in self._processed_document_hashes:
            logging.info(f"Hash match. Document {document_processor.file_path} already in Vector Store")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=document_type.chunk_config["size"],
                                                       chunk_overlap=document_type.chunk_config["overlap"])

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self._text_embeddings,
            persist_directory=self._persistent_path
        )
        logging.info(f"Initialize collection {collection_name}")

        document_loader = document_processor.load()
        all_chunks = []
        for each_doc_collection in document_loader:
            for each_document in each_doc_collection:
                # Process one page at a time (e.g., split and add to ChromaDB)
                chunks = text_splitter.split_documents([each_document])
                all_chunks.extend(chunks)

        # Using a batch_size of 100 provides the best balance for 16GB RAM
        batch_size = 100
        num_chunks = len(all_chunks)

        logging.info(f"Adding {num_chunks} chunks to Chroma in batches of {batch_size}...")

        for i in range(0, num_chunks, batch_size):
            batch = all_chunks[i: i + batch_size]
            vectorstore.add_documents(documents=batch)
            logging.info(f"Progress: {min(i + batch_size, num_chunks)}/{num_chunks} chunks indexed.")

        self._collections[collection_name] = vectorstore
        self._processed_document_hashes.add(file_hash)

    def retrieve(self, query: str, k: int = 4) -> str:
        """
        Retrieves the top 'k' most relevant chunks from a specific collection.

        Args:
            query: The user's question.
            k: Number of chunks to retrieve.
        """
        context_parts = []

        for each_collection_name in self._collections.keys():
            logging.info(f"Searching collection {each_collection_name}")
            docs = self._collections[each_collection_name].similarity_search(query, k=k)

            for doc in docs:
                page_num = doc.metadata.get("page", "unknown")
                content = doc.page_content.replace("\n", " ")
                context_parts.append(f"[Source: Document {each_collection_name} Page {page_num}] {content}")

        return "\n\n".join(context_parts)

