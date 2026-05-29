import pymupdf
import os
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np

# parse pdfs into plain text, return each page of plain text in a list for one pdf
def returnParsedPDFText(filePath: str, ocr:PaddleOCR, dpi: int, skipTitle: bool):
    try:
        doc = pymupdf.open(f"{filePath}")
    except Exception as e:
        print(f"Error opening {filePath}: {e}\nSkipping this file.\n")
        return []
    pages = []
    blockid = 0
    for i, page in enumerate(doc):
        if (i == 0 or i == len(doc)-1) and skipTitle: #heuristic for our own slides. skips title slides and (mostly) ending slides
            continue
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
                try:
                    result = ocr.predict(imgnp)
                except Exception as e:
                    print(f"Error trying to OCR {filePath}: ", e)

                ocrWords = "\n".join(result[0]["rec_texts"])
                if not ocrWords or ocrWords.isspace():
                    continue
                pageText += "\n" + ocrWords + "\n"
                blockid += 1
            else:
                pageText += block[4]

        pages.append(pageText)
    return pages

"""
Parse a single file into chunks, and return them along with metadatas and ids.
"""
def formFileChunks(root: str, file: str, ocr:PaddleOCR | None, dpi: int, id: int):
    chunks, ids, metadata = [], [], []
    parsed = False
    if file.endswith(".pdf"):
        skipTitle = False
        if "Handout" not in file: skipTitle = True
        pages = returnParsedPDFText(f"{root}/{file}", ocr, dpi, skipTitle)
        pageNum = 0
        for p in pages:
            if not p.strip() or len(p) == 0:
                pageNum += 1
                continue
            parsed = True
            meta = {"fileName": file, "page": pageNum, "fileType": "pdf", "id": str(id)} #redundant id for filtering
            ids.append(str(id))
            chunks.append(p)
            metadata.append(meta)
            pageNum += 1
            id += 1
    elif file.endswith(".cpp") or file.endswith(".py") or file.endswith(".cc"):
        with open(f"{root}/{file}", "r") as f:
            code = f.read()
            if code.strip() and len(code) != 0:
                parsed = True
                meta = {"fileName": file, "page": 0, "fileType": Path(file).suffix, "id": str(id)} #redundant id for filtering
                ids.append(str(id))
                chunks.append(code)
                metadata.append(meta)
                id += 1

    return chunks, ids, metadata, id, parsed


"""
Walk a directory to parse its contents. Uses a generator function to avoid keeping large results entirely in RAM
Input: A directory path.
Output: documents: list[str] -> a list of text chunks. If from a pdf, each is a page. If code, it is the entire file.
        ids: list[int] -> a list of ids, they are unique to each chunk.
        metadata: list[dict[str:Any]] -> a list of metadata for each chunk formatted as so:
        meta = {"fileName": file, "page": pageNum, or 0 if code, "fileType": .fileextension}
"""
def walkParse(inputDirPath: str, startId: int):
    ocr = PaddleOCR(lang = 'en', use_angle_cls = True, device = "gpu") #move to an argument eventually, but this is intended to parse the materials all at once, so no reinstantiation.
    id = startId
    for root, dirs, files in os.walk(inputDirPath):
        for file in files:
            chunks, ids, metadata, newId, parsed = formFileChunks(root, file, ocr, 300, id)
            if parsed:
                id = newId
                yield chunks, ids, metadata, id, file
            else:
                yield [], [], [], -1, file