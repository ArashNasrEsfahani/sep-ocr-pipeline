# SHL OCR Program: Consolidated Tasks 1-4

This repository contains the complete OCR pipeline developed for the SHL Challenge, progressing through four staged tasks from text detection to attention-based recognition on Persian receipts.

## Overview

The SHL OCR initiative covers:
- **Task 1**: Detection baseline using YOLOv5/YOLOv8
- **Task 2**: Recognition baseline with PaddleOCR PP-OCRv4  
- **Task 3**: Detection fine-tuning on Arshasb dataset
- **Task 4**: TrOCR fine-tuning for sequence-to-sequence recognition

## Project Structure

```
ocr/
├── task1_detection/           # YOLOv5/YOLOv8 detection experiments
├── task2_text_extraction/     # PaddleOCR baseline recognition
├── task3_detection_finetune/  # Arshasb fine-tuning experiments  
├── task4_text_recognition_finetune/  # TrOCR attention-based recognition
├── figures/                   # Report figures and visualizations
├── dataSample/               # Sample receipt images (not tracked)
└── report_tasks1_4.tex       # Comprehensive LaTeX report
```

## Key Findings

- **Pre-trained PaddleOCR outperformed fine-tuned models** across CER, WER, and exact-match metrics
- **Arshasb fine-tuning provided minimal transfer** to Parsynth-style receipts  
- **Multi-receipt inputs remain challenging**, causing overlapping detections and recognition errors
- **Broad multilingual pre-training generalizes well** to Persian receipts without domain-specific fine-tuning

## Requirements

The project uses Python 3.10 with key dependencies:
- `transformers==4.44.2`
- `accelerate==0.33.0` 
- `paddlepaddle` and `paddleocr`
- `ultralytics` (YOLOv8)
- `torch`, `torchvision`

## Usage

Each task directory contains:
- `scripts/` - Training and evaluation scripts
- `artifacts/` - Results, metrics, and model outputs
- `requirements_task*.txt` - Task-specific dependencies

See individual task READMEs for detailed instructions.

## Results

| Model | CER | WER | EM |
|-------|-----|-----|-----|
| PaddleOCR Baseline | 0.418 | 0.723 | 0.302 |
| TrOCR V2 (frozen) | 0.999 | 1.104 | 0.000 |

## Documentation

The complete technical report is available in `report_tasks1_4.tex`, covering:
- Model selection rationale
- Experimental configurations  
- Quantitative and qualitative results
- Cross-task integration lessons
- Future development roadmap

## Contributing

This repository documents completed research. For questions or collaboration, please refer to the technical report or contact the SHL OCR Team.
