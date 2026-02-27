import torch
from docling.models.stages.layout.layout_model import LayoutModel
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import LayoutOptions
from paddleocr import LayoutDetection
from paddleocr import DocImgOrientationClassification
from pymupdf import Document
import utils, io
from config import PDF_DPI, WORKER_BATCHSIZE, ROTATION_PREDICTION_THRESHOLD
from PIL import Image
from typing import List
import numpy as np

from models.page_model import PageModel, PageLayout


class DocumentProcessor:

    MOBILE_NET_ROTATION_CLASS = {
        0: 0,
        1: 90,
        2: 180,
        3: 270,
    }

    def __init__(self):
        # Get the device to run on
        self.torch_device = utils.get_torch_device()

        # Initialize the model
        self.model_block_detection = LayoutDetection(model_name="PP-DocBlockLayout")
        self.model_image_rotation_detection = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
        self.layout_model = LayoutModel(None, AcceleratorOptions(), LayoutOptions())

    # Crop the content of the page by using BOX detection model, that detects the document blocks
    def crop_content(self, imgs: List[Image.Image]) -> List[Image.Image | None]:
        np_array = [np.asarray(img) for img in imgs]
        block_det_res_boxes = self.model_block_detection.predict(np_array, batch_size=WORKER_BATCHSIZE)
        results = []
        for im, box_res in zip(imgs, block_det_res_boxes):
            boxes = box_res['boxes']
            block_det_res_boxes = [b for b in boxes if b['label'] == 'Region']
            if block_det_res_boxes:
                block_det_res_boxes = sorted(block_det_res_boxes, key=lambda x: x['score'], reverse=True)
                x1, y1, x2, y2 = block_det_res_boxes[0]['coordinate']
                results.append(im.crop((x1, y1, x2, y2)))
            else:
                results.append(None)
        return results

    def get_page_headers(page: dict) -> int | None:
        layouts = page['layout']
        page_header_y = [layout['b'] for layout in layouts if layout['label'] in ['Page-header']]
        if page_header_y:
            return max(page_header_y)
        return None


    # Get the footer of the page
    def get_page_footer(page: dict) -> int | None:
        layouts = page['layout']
        page_footer_y = [layout['b'] for layout in layouts if layout['label'] in ['Page-footer']]
        if page_footer_y:
            return min(page_footer_y)
        return None

    # Get the rotation of the page
    def get_image_rotation_detection(self,imgs: List[Image.Image]) -> List[dict[str, float]]:
        imgs = [np.asarray(im) for im in imgs]
        outputs = self.model_image_rotation_detection.predict(input=imgs, batch_size=WORKER_BATCHSIZE)
        results = [{"class_id": o['class_ids'][0], 'label_name': o['label_names'][0], 'score': o['scores'][0]} for o in
                   outputs]
        return results

    # Process the document
    async def process_page(self, document: Document)-> List[PageModel]:
        total_pages = document.page_count

        processed_pages = []

        #Process the pages in batches
        for start_page in range(0, total_pages, WORKER_BATCHSIZE):
            end_page = start_page + WORKER_BATCHSIZE
            if start_page > total_pages:
                break
            if end_page > total_pages:
                end_page = total_pages
            pages = document.pages(start_page, end_page)

            pages_images = [page.get_pixmap(dpi=PDF_DPI).pil_image() for page in pages]

            # Get a document box for rotation detection
            pages_croped = self.crop_content(pages_images)

            # Estimate the rotation of the page from croped content
            rotation_results = self.get_image_rotation_detection(pages_croped)
            rotations = []
            for idx, rot in enumerate(rotation_results):
                cls_idx = rot['class_id']
                cls = int(rot['label_name'])
                conf = rot['score']
                im = pages_images[idx]

                # If the confidence is high enough, rotate the image
                if conf > ROTATION_PREDICTION_THRESHOLD:
                    im = im.rotate(cls, expand=True)

                pages_images[idx] = im
                rotations.append(cls)

            # Predict the layout of the page
            predicted_page_layouts = self.layout_model.layout_predictor.predict_batch(pages_images)
            for ix, (rotation, layouts, image) in enumerate(zip(rotations, predicted_page_layouts, pages_images)):
                page_no = start_page + ix
                page_layouts = []
                im_w, im_h = image.size
                content_y1 = 0
                content_y2 = im_h
                page_pictures = []
                for layout in layouts:
                    if layout['label'] == 'Page-header':
                        # Crop the header area
                        content_y1 = max(content_y1, layout['b'])
                    elif layout['label'] == 'Page-footer':
                        # Crop the footer area
                        content_y2 = min(content_y2, layout['t'])
                    elif layout['label'] == 'Picture':
                        # Crop the picture area
                        pic = image.crop((layout['l'], layout['t'], layout['r'], layout['b']))
                        page_pictures.append(pic)

                    page_layout = PageLayout(label=layout['label'],
                                         top=layout['t'],
                                         right=layout['r'],
                                         bottom=layout['b'],
                                         left=layout['l'])

                    page_layouts.append(page_layout)

                # Crop the content area
                header_image = None
                footer_image = None
                if content_y1 > 0:
                    header_image = image.crop( (0, 0, im_w, content_y1) )
                    image = image.crop( (0, content_y1, im_w, im_h) )
                    im_w, im_h = image.size

                    if content_y2 < im_h:
                        content_y2 = max(content_y2 - content_y1, 0)

                if 0 < content_y2 < im_h:
                    footer_image = image.crop( (0, content_y2, im_w, im_h) )

                page = PageModel(rotation=rotation,
                                 page_layout=page_layouts,
                                 page_footer=footer_image,
                                 page_header=header_image,
                                 page_content=image,
                                 page_pictures=page_pictures,
                                 page_no=page_no)
                processed_pages.append(page)
        return processed_pages