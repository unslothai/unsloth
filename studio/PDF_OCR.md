# Scanned PDF uploads

Studio can extract text from scanned PDFs in Chat, project sources, and Data Recipes.
Existing selectable text is preserved. OCR runs on image-only pages, including scans
with a selectable header or footer. Small logos do not require OCR when selectable
text is present. Large images with little selectable body text are treated as
possible scans, so illustrated pages can also require OCR. Chat has a separate
figure-captioning setting.

Chat first uses the loaded local vision GGUF model when available, then tries local
Tesseract OCR for scanned pages that were not transcribed. Data Recipes uses local
Tesseract OCR without requiring a loaded chat model.
Recipe PDF extraction runs in separate worker processes so OCR does not block
other requests.

## Local OCR setup

PyMuPDF includes the OCR integration. Install Tesseract language data on the machine
running the Studio backend, and set TESSDATA_PREFIX to the folder containing the
.traineddata files before starting Studio. No data is downloaded automatically.

For example, install your operating system's Tesseract package and English language
data, then point TESSDATA_PREFIX to its tessdata directory. You can also obtain
language files from the [official Tesseract repository](https://github.com/tesseract-ocr/tessdata_fast).
See [PyMuPDF's OCR setup](https://pymupdf.readthedocs.io/en/latest/installation.html#enabling-integrated-ocr-support)
for platform-specific details.

Set RAG_OCR_LANGUAGE to the installed language codes; the default is eng.
For example, eng+deu requires both eng.traineddata and deu.traineddata.

## Limits and failed uploads

RAG_OCR_SCANNED=0 disables scanned-page OCR by default. Chat's **OCR scanned pages**
setting overrides this for chat/project uploads. Data Recipes follows the backend
default.

RAG_OCR_MAX_PAGES defaults to 20 scanned pages per PDF. Raise it before starting
Studio when longer scans are expected. Local OCR uses RAG_OCR_DPI (default 150).

If scanned pages remain unreadable, the upload fails with their page numbers rather
than silently accepting an incomplete document. Configure the OCR language data,
enable OCR, increase the page budget when necessary, or upload a searchable PDF,
then attach the file again. Empty documents also fail instead of appearing indexed
with zero searchable chunks.

OCR is fallible, especially for handwriting, low-resolution scans, and complex
tables. Check extracted results against the original document.
