## OCR Service Example

This is a simple example of an OCR service by using the Vision Language Model cooperative with document analysis models.

In this example it focused on adaptation of the [TyphoonOCR](https://huggingface.co/typhoon-ai/typhoon-ocr1.5-2b) model as a robust engine.

For prevention of producing llm hallucinations, we use a combination of document analysis models and image rotation detection to ensure accurate and reliable OCR results.
1. LayoutDetection we use [PP-DocBlockLayout](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/configs/det/PP-DocBlockLayout.yml) for detecting document blocks for better document orientation detection.
2. DocImgOrientationClassification we use [PP-LCNet_x1_0_doc_ori](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/configs/classification/PP-LCNet_x1_0_doc_ori.yml) for detecting image rotation.
3. For layout detection we use docling LayoutModel for detect image, header, footer and text.
4. With layout detection, we can ensure the image for OCR has content to process. (if the image is empty, model will reproduce tons of hallucinations/ new lines)

Please refer to the [model card](https://huggingface.co/typhoon-ai/typhoon-ocr1.5-2b) for more details.
For running this example, you can use uvicorn to run the server.

```bash
uvicorn service:app --reload
```

For LLM service, you can run ollama or lmdeploy to run TyphoonOCR model for local inference.
For more details, please refer to the [Typhoon-OCR1.5-2b-GGUF](https://huggingface.co/aonaon/typhoon-ocr1.5-2b-gguf).

For configuring the service, you can copy config.example.yaml to config.yaml and modify the configuration.
