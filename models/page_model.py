from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from PIL import Image
import dataclasses

class PageLayout(BaseModel):
    label: str = ""
    top: float
    left: float
    bottom: float
    right: float

@dataclass
class PageModel:
    page_no: int
    rotation: int
    page_layout: List[PageLayout]
    page_content: Image.Image
    page_footer: Optional[Image.Image] = None
    page_header: Optional[Image.Image] = None
    page_pictures: List[Image.Image] = dataclasses.field(default_factory=lambda: [])

class OCRPageModel(PageModel):
    footer_text: str = ""
    content_text: str = ""
    header_text: str = ""

class OCRPageResultModel(BaseModel):
    page_no: int
    footer_text: str = ""
    content_text: str = ""
    header_text: str = ""
    page_pictures: List[str] = Field(default=[], description="Page images in base64 format")