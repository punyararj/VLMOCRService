## BaseText is Context for hint the ocr

PROMPTS = {
    "default": lambda base_text: (f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"),
    "structure": lambda base_text: (
        f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
    "no_hint":
        f"Below is an image of a document page. "
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n",

}
WORKER_BATCHSIZE=4
PDF_DPI = 300
OCR_BASE_URL='https://xxxx/llm/v1'
OCR_API_KEY=''
OCR_MODEL='scb10x/typhoon-ocr1.5-2b'
OCR_TOP_P=0.6
OCR_TEMPERATURE=0.1
OCR_TIMEOUT=120
ROTATION_PREDICTION_THRESHOLD=0.4
OCR_EXTRABODY={
              "repetition_penalty": 1.2,
          }