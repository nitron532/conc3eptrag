import pymupdf
import os
from pathlib import Path

#parse pdfs into plain text, return each page of plain text in a list for one pdf
def returnParsedPDFText(filePath: str):
    doc = pymupdf.open(f"{filePath}")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return pages


"""
Walk a directory to parse its contents.
Input: A directory path.
Output: documents: list[str] -> a list of text chunks. If from a pdf, each is a page. If code, it is the entire file.
        ids: list[int] -> a list of ids, they are unique to each chunk.
        metadata: list[dict[str:Any]] -> a list of metadata for each chunk formatted as so:
        meta = {"fileName": file, "week": parentdir/filename, "page": pageNum, or 0 if code, "fileType": .fileextension}
"""
def walkParse(inputDirPath: str):
    chunks = []
    ids = []
    metadata = []
    id = 0
    for root, dirs, files in os.walk(inputDirPath):
        for file in files:
            if file.endswith(".pdf"):
                pages = returnParsedPDFText(f"{root}/{file}")
                pageNum = 0
                for p in pages:
                    if len(p) == 0:
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
                    if len(code) == 0: continue
                    meta = {"fileName": file, "week": root[root.rfind("/"):], "page": 0, "fileType": Path(file).suffix}
                    ids.append(str(id))
                    chunks.append(code)
                    metadata.append(meta)
                    id += 1
            
    return chunks, ids, metadata