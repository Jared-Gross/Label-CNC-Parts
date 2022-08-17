# Embed-CNC-Part-List

Labels parts with their unit number and short letter abreviation.

**BEFORE**
![image](https://user-images.githubusercontent.com/25397800/185258203-48304f07-a071-41b1-af98-be348081a9fe.png)

**AFTER**
![image](https://user-images.githubusercontent.com/25397800/185258108-5ec319ad-2d65-4afe-8f64-387f29a7d16b.png)

## Requirements

### Python requirements:
```
pip install -r requirements.txt
```

### Third party software:

Download [Poppler](https://blog.alivate.com.au/poppler-windows/) Or [Follow these instructions](https://pdf2image.readthedocs.io/en/latest/installation.html#windows)

Download and install [pytesseract](https://www.softpedia.com/get/Programming/Other-Programming-Files/Tesseract-OCR.shtml) 

## Build

No build is available, i don't know if one is even possible, as `pyinstaller` and `opencv-python` have __poor chemistry__.

## Usage

To run this program as user friendly as possible, I suggest following the instructions from my [other repository on context menus](https://github.com/JareBear12418/The-BEST-way-to-convert-files) and edit and use the `run.bat` file.
