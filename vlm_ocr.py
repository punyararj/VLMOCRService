import asyncio
from typing import List

from paddlex.inference.pipelines.layout_parsing.merge_table import is_skippable

import utils, json
import config
from config import *
from PIL.Image import Image
from openai.types.chat import ChatCompletionUserMessageParam, \
     ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai import OpenAI, AsyncOpenAI
from models.page_model import PageModel, OCRPageModel, OCRPageResultModel


class OCRService:

    def __init__(self):
        self.async_openai_client = AsyncOpenAI(base_url=OCR_BASE_URL, api_key=OCR_API_KEY)

    async def do_vlm_ocr(self, image:Image, hint_text:str=None, prompt='no_hint', max_tokens = 10000)->str:
        if image is None:
            return ''
        if hint_text is None or hint_text == 'no_hint':
            prompt = PROMPTS['no_hint']
        else:
            prompt = PROMPTS[prompt](hint_text)
        image_url = utils.get_url_base64_str(image)
        messages = [
            ChatCompletionUserMessageParam(content=[
                ChatCompletionContentPartTextParam(text=prompt, type="text"),
                ChatCompletionContentPartImageParam(image_url=ImageURL(url=image_url), type="image_url"),
            ], role="user")]

        results = await self.async_openai_client.chat.completions.create(messages=messages,
                                                        model=OCR_MODEL,
                                                        temperature=OCR_TEMPERATURE,
                                                        top_p=OCR_TOP_P,
                                                        extra_body=OCR_EXTRABODY,
                                                        timeout=OCR_TIMEOUT,
                                                        max_tokens=max_tokens)
        return results.choices[0].message.content

    def decode_ocr_result(self, text:str)->str:
        try:
            json_obj = json.loads(text)
            return json_obj['natural_text']
        except Exception as e:
            return text

    async def page_ocr(self, content_image:Image, header_image:Image,
                       footer_image:Image, hint_text:str=None)->tuple[str, str, str]:
        if hint_text is None:
            #tasks = [self.do_vlm_ocr(content_image),]
            tasks = [self.do_vlm_ocr(content_image),
                     self.do_vlm_ocr(header_image),
                     self.do_vlm_ocr(footer_image)]
        else:
            tasks = [self.do_vlm_ocr(content_image, hint_text, 'structure'),
                     self.do_vlm_ocr(header_image),
                     self.do_vlm_ocr(footer_image)]
        results = await asyncio.gather(*tasks)
        #results = [results[0],'','']
        results = [self.decode_ocr_result(r) if r!='' else '' for r in results]
        return results[0], results[1], results[2]

    # Process the pages and return the result model
    # This method is needs to refactor
    async def page_process_no_ref(self, pages:List[PageModel],
                                  batch_size:int=4) -> List[OCRPageResultModel]:

        ocr_results = []

        # Process the pages in batches
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            tasks = []
            ocr_pages = []
            for pi, page in enumerate(batch):
                # Skip the page if it is not ocrable, no content to ocr
                is_ocrable = len([l for l in page.page_layout if l.label not in ['Page-header', 'Page-footer']])>0
                if is_ocrable:
                    ocr_pages.append(pi)
                    tasks.append(self.page_ocr(page.page_content, page.page_header, page.page_footer))

            # Execute the tasks concurrently
            results = await asyncio.gather(*tasks)
            for bx, (content_text, header_text, footer_text)  in enumerate(results):
                result_id = ocr_pages[bx]
                pictures = batch[result_id].page_pictures
                b64_pictures = [utils.get_url_base64_str(p) for p in pictures]

                # Save result to a result model
                ocr_page = OCRPageResultModel(page_no=batch[result_id].page_no,
                                              content_text=content_text,
                                              header_text=header_text,
                                              footer_text=footer_text,
                                              page_pictures=b64_pictures)
                ocr_results.append(ocr_page)
        return ocr_results
