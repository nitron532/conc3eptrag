import pymupdf
import os
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np

# parse pdfs into plain text, return each page of plain text in a list for one pdf
def returnParsedPDFText(filePath: str, ocr:PaddleOCR, dpi: int, imageid: str):
    doc = pymupdf.open(f"{filePath}")
    pages = []
    blockid = 0
    for page in doc:
        pageBlocks = page.get_text("blocks", flags = pymupdf.TEXTFLAGS_BLOCKS | pymupdf.TEXT_PRESERVE_IMAGES)
        pageText = ""
        width = page.rect.width
        height = page.rect.height
        for block in pageBlocks:
            if (block[-1] == 1 or block[-1] == 3): # image detected
                if (round(block[2]) - round(block[0]) == width and (round(block[3]) - round(block[1]) == height)): #entire page layout from pptx style pdfs
                    continue
                blockRect = pymupdf.Rect(block[0],block[1],block[2],block[3])
                if len(page.get_text("words", clip = blockRect)) > 0: #if text was overlaid on an image, skip
                    continue

                scale = dpi / 72  # for bounding box coordinate scaling, pymupdf uses 72 dpi
                rect = pymupdf.Rect(block[0], block[1], block[2], block[3])
                mat = pymupdf.Matrix(scale, scale)  # scale matrix at 300 DPI
                pmap = page.get_pixmap(matrix = mat, clip = rect, dpi = dpi)
                pil = Image.frombytes("RGB", [pmap.width, pmap.height], pmap.samples)

                imgnp = np.array(pil)
                result = ocr.predict(imgnp)

                ocrWords = "\n".join(result[0]["rec_texts"])
                if not ocrWords or ocrWords.isspace():
                    continue
                pageText += "\n" + ocrWords + "\n"
                pil.save(f"../ocrdimages/{imageid} {blockid}.png", "PNG") # uncomment to verify cropped OCR sections
                blockid += 1
            else:
                pageText += block[4]

        pages.append(pageText)
    return pages

"""
Walk a directory to parse its contents. Uses a generator function to avoid keeping large results entirely in RAM (and possibly partly in disk D:)
Input: A directory path.
Output: documents: list[str] -> a list of text chunks. If from a pdf, each is a page. If code, it is the entire file.
        ids: list[int] -> a list of ids, they are unique to each chunk.
        metadata: list[dict[str:Any]] -> a list of metadata for each chunk formatted as so:
        meta = {"fileName": file, "week": parentdir/filename, "page": pageNum, or 0 if code, "fileType": .fileextension}
"""
def walkParse(inputDirPath: str, startId: int):
    ocr = PaddleOCR(lang = 'en', use_angle_cls = True, device = "gpu") #move to an argument eventually, but this is intended to parse the materials all at once, so no reinstantiation.
    chunks, ids, metadata = [], [], []
    id = startId
    for root, dirs, files in os.walk(inputDirPath):
        for file in files:
            parsed = False
            if file.endswith(".pdf"):
                pages = returnParsedPDFText(f"{root}/{file}", ocr, 300, file)
                parsed = True
                pageNum = 0
                for p in pages:
                    if not p.strip() or len(p) == 0:
                        pageNum += 1
                        continue
                    meta = {"fileName": file, "week": root[root.rfind("/")+1:], "page": pageNum, "fileType": "pdf"}
                    ids.append(str(id))
                    chunks.append(p)
                    metadata.append(meta)
                    pageNum += 1
                    id += 1
            elif file.endswith(".cpp") or file.endswith(".py") or file.endswith(".cc"):
                with open(f"{root}/{file}", "r") as f:
                    code = f.read()
                    if not code.strip() or len(code) == 0: continue
                    parsed = True
                    meta = {"fileName": file, "week": root[root.rfind("/"):], "page": 0, "fileType": Path(file).suffix}
                    ids.append(str(id))
                    chunks.append(code)
                    metadata.append(meta)
                    id += 1
            if parsed:
                yield chunks, ids, metadata, id
                chunks, ids, metadata = [], [], []