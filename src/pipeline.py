"""
Old Documents Cleanup & OCR Pipeline
------------------------------------
Reusable functions for the stages demonstrated in
notebooks/old_documents_pipeline.ipynb. Import these into your own
scripts for batch processing instead of running the notebook cell by cell.

Example
-------
    from src.pipeline import clean_document, extract_text

    cleaned = clean_document("sample_data/old_document.png")
    text = extract_text(cleaned)
"""

import cv2
import numpy as np
import pytesseract


def load_grayscale(image_path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Remove speckle/scan noise while preserving text edges."""
    return cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)


def get_skew_angle(image: np.ndarray, angle_range: float = 5.0, step: float = 0.1) -> float:
    """
    Detect skew angle using the projection-profile method: rotate a
    binarized copy through a range of angles and pick the one whose
    horizontal row-sum profile has the highest variance (i.e. text lines
    are most sharply aligned).
    """
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    (h, w) = thresh.shape
    center = (w // 2, h // 2)

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-angle_range, angle_range + step, step):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        row_sums = rotated.sum(axis=1).astype(np.float64)
        score = row_sums.var()
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by the given angle (from get_skew_angle) to straighten it."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Boost local contrast with CLAHE, useful for faded/unevenly aged paper."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray: np.ndarray, block_size: int = 25, c: int = 15) -> np.ndarray:
    """Adaptive thresholding to convert to clean black-and-white text."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )


def remove_small_blobs(binary: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Remove small leftover stain/noise specks via connected-component filtering."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary, connectivity=8)
    mask = np.ones_like(binary) * 255
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 0
    return mask


def clean_document(image_path: str) -> np.ndarray:
    """Run the full cleanup pipeline on an image path and return the cleaned binary image."""
    gray = load_grayscale(image_path)
    denoised = denoise(gray)
    angle = get_skew_angle(denoised)
    deskewed = deskew(denoised, angle)
    contrast = enhance_contrast(deskewed)
    binary = binarize(contrast)
    cleaned = remove_small_blobs(binary)
    return cleaned


def extract_text(cleaned_image: np.ndarray, lang: str = "eng") -> str:
    """Run Tesseract OCR on a cleaned image."""
    config = r"--oem 3 --psm 6"
    return pytesseract.image_to_string(cleaned_image, config=config, lang=lang)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <image_path>")
        sys.exit(1)

    cleaned = clean_document(sys.argv[1])
    text = extract_text(cleaned)
    print(text)
