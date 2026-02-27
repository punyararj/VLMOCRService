"""Containers module."""

from dependency_injector import containers, providers
from vlm_ocr import OCRService
from document_processor import DocumentProcessor

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    vllm_ocr = providers.Factory(OCRService)
    document_processor = providers.Singleton(DocumentProcessor)