from fastapi.middleware.cors import CORSMiddleware
from dependency_injector.wiring import inject, Provide
from fastapi import FastAPI, Depends, File, HTTPException, status, Request, Response, UploadFile, Header
from fastapi.responses import JSONResponse, FileResponse
from pymupdf import pymupdf

from config import WORKER_BATCHSIZE
from containers import Container
from document_processor import DocumentProcessor
from models.APIResponse import ApiResponse
from models.page_model import OCRPageResultModel
from typing import List
import asyncio

from vlm_ocr import OCRService


class APIException(Exception):
    def __init__(self, name: str, http_status: int = 500):
        self.name = name
        self.status = http_status

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(APIException)
async def unicorn_exception_handler(request: Request, exc: APIException):
    resp = ApiResponse()
    resp.status = exc.status
    resp.message = exc.name
    return JSONResponse(
        status_code=exc.status,
        content=resp.dict(),
    )


def read_pdf_content(uploaded_file):
    if uploaded_file is not None:
        # Read the file content as bytes
        file_bytes = uploaded_file.read()

        # IMPORTANT: If using st.file_uploader, you may need to seek to the beginning
        # if the file has already been read once in the same script run.
        uploaded_file.seek(0)

        # Open the document from the bytes stream, specifying the file type
        # The filetype parameter helps PyMuPDF determine the document format
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()  # Extract text from each page
            doc.close()
            return text
        except Exception as e:
            raise APIException(f"Error reading PDF: {e}")
    return None


@app.api_route("/process_doc", response_model=ApiResponse[List[OCRPageResultModel]])
@inject
async def process_file(uploaded_file: UploadFile = File(...),
                       document_processor:DocumentProcessor = Depends(Provide[Container.document_processor]),
                       ocr_engine:OCRService = Depends(Provide[Container.vllm_ocr])):

    if uploaded_file.filename.split('.')[-1] != 'pdf':
        raise APIException("File must be a PDF", status.HTTP_400_BAD_REQUEST)

    file_bytes = uploaded_file.read()

    # IMPORTANT: If using st.file_uploader, you may need to seek to the beginning
    # if the file has already been read once in the same script run.
    uploaded_file.seek(0)
    pdf_doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    result = asyncio.run(document_processor.process_page(pdf_doc))
    ocr_results = asyncio.run(ocr_engine.page_process_no_ref(result, batch_size=WORKER_BATCHSIZE))
    return ocr_results

container = Container()
container.wire(modules=[__name__])
