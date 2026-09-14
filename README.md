# Historic Documents Cleanup

A from-scratch image-processing pipeline for cleaning up scans of old/damaged historic documents and extracting their text via OCR, built using OpenCV, Tesseract, and Jupyter Notebook. This is a project built as part of the coursework for Image Processing and its Applications.

## Overview

The Historic Documents Cleanup pipeline takes a scanned or photographed page of an aged/damaged document and progressively cleans it up; removing noise, straightening skewed scans, boosting faded contrast, and binarizing the image — before passing it to Tesseract for text extraction. Each stage is visualized step by step so you can see exactly what it's doing to the image, and every parameter is tunable for different scan qualities and font sizes.

## Features

* Non-Local Means denoising to remove speckle/scan noise while preserving text edges
* Projection-profile deskew — robust to stains and irregular paragraph shapes
* CLAHE contrast enhancement for faded, unevenly aged paper
* Adaptive-threshold binarization for clean black-and-white text
* Optional morphological cleanup, with a built-in before/after OCR comparison so you can verify it actually helps before trusting it
* Automatic upscaling for low-resolution scans, where thin letter strokes would otherwise be lost
* OCR text extraction via Tesseract
* Both a visual notebook walkthrough and a reusable Python module for scripting/batch use

## Technology Stack

* Python
* OpenCV
* NumPy
* Pillow (PIL)
* Tesseract OCR / pytesseract
* Matplotlib
* Jupyter Notebook

## Architecture

```text
Scanned Document
    ↓
Grayscale Conversion
    ↓
Denoising (Non-Local Means)
    ↓
Deskew (Projection-Profile Method)
    ↓
Upscale (for low-resolution scans)
    ↓
Contrast Enhancement (CLAHE)
    ↓
Binarization (Adaptive Threshold)
    ↓
Morphological Cleanup (optional)
    ↓
Tesseract OCR
    ↓
Extracted Text
```

## How It Works

1. The scanned image is converted to grayscale, dropping color information that isn't needed for text cleanup.
2. Non-Local Means denoising removes speckle and paper-grain noise while preserving text edges.
3. A projection-profile method detects and corrects skew: a binarized copy is rotated through a range of angles, and the angle producing the sharpest horizontal row-sum profile is selected as the true skew.
4. If the source image is low-resolution relative to its font size, it's upscaled to give the remaining steps enough pixels to work with.
5. CLAHE boosts local contrast on faded or unevenly aged paper without blowing out already-bright regions.
6. Adaptive thresholding binarizes the image into clean black-and-white text, handling uneven lighting/aging better than a single global threshold.
7. An optional morphological cleanup step can remove small leftover stain/noise specks — but since this can also delete real letter details on small or serif fonts, the notebook runs OCR with and without it side by side so you can decide for your own images.
8. Tesseract extracts the final text from the cleaned image.

This step-by-step structure means any stage can be swapped, tuned, or skipped for a given document without touching the rest of the pipeline.

## Project Structure

```text
historic-documents-cleanup/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── historic_documents_cleanup.ipynb
├── src/
│   └── pipeline.py
├── sample_data/
│   └── old_document.png
└── output/
```

* `notebooks/historic_documents_cleanup.ipynb`: step-by-step walkthrough with visualizations at every stage
* `src/pipeline.py`: reusable functions for scripting/batch use
* `sample_data/old_document.png`: synthetic aged-document sample for demo
* `output/`: cleaned images and extracted text land here

## Setup

```bash
git clone <your-repo-url>
cd historic-documents-cleanup
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need the Tesseract OCR engine installed on your system (separate
from the Python package):

* **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
* **macOS**: `brew install tesseract`
* **Windows**: install from the [official Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### Option A — Notebook (recommended for exploring/visualizing each step)

```bash
jupyter notebook notebooks/historic_documents_cleanup.ipynb
```

Swap the `IMAGE_PATH` variable in the "Load Image" cell to point at your own scanned document.

### Option B — Script / batch processing

```bash
python3 src/pipeline.py sample_data/old_document.png
```

Or import the functions directly:

```python
from src.pipeline import clean_document, extract_text

cleaned = clean_document("sample_data/old_document.png")
text = extract_text(cleaned)
print(text)
```

## Next Steps / Ideas to Extend

* **Batch processing**: loop `clean_document` + `extract_text` over a folder of scans
* **Better deskew**: for curved/book-spine scans, a Hough-line-based approach may help
* **Super-resolution**: a learned upscaling model before OCR for very low-DPI scans
* **Layout detection**: use `pytesseract.image_to_data` with `psm 3` to preserve reading order on multi-column documents
* **Multi-language OCR**: pass `lang="fra"` (etc.) to `extract_text` (requires the matching `tesseract-ocr-<lang>` package)

## Note

Pipeline parameters (denoising strength, block sizes, morphological cleanup) were tuned empirically and can vary in effectiveness across document types, font sizes, and scan resolutions — always compare OCR output before and after a step on your own images rather than assuming a default is optimal.
