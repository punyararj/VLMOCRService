import unittest, asyncio
import pymupdf

import vlm_ocr
from document_processor import DocumentProcessor

DOC_PATH = 'TOR.pdf'
class MyTestCase(unittest.TestCase):

    def test_proces_page(self):
        pdf_doc = pymupdf.Document(DOC_PATH)
        document_processor = DocumentProcessor()
        result = asyncio.run(document_processor.process_page(pdf_doc))
        '''for rs in result:
            rs.page_content.save(f'./tmp/content_{rs.page_no}.png')
            if rs.page_header:
                rs.page_header.save(f'./tmp/header_{rs.page_no}.png')
            if rs.page_footer:
                rs.page_footer.save(f'./tmp/footer_{rs.page_no}.png')'''
        ocr_engine = vlm_ocr.Service()
        ocr_results = asyncio.run(ocr_engine.page_process_no_ref(result, batch_size=4))
        print(ocr_results)
        #assert(len(ocr_results) > 0)
        #self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
