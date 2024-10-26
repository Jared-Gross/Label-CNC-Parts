import contextlib
import io
import json
import os
import re
import shutil
import sys
import webbrowser
from operator import attrgetter
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from natsort import natsorted
from pdf2image import convert_from_path
from PIL import Image
from rich import print

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR/tesseract.exe"

program_directory = os.path.dirname(os.path.realpath(sys.argv[0]))
with open(f"{program_directory}/part_names.json") as part_names_file:
    part_names = json.load(part_names_file)

file_info = {}

PART_INFO_REGEX = r"(\d{1,})  =  .{1,}_([0-9]+)_(.{1,})"
OTHER_PART_INFO_REGEX = r"(\d{1,})  =  ([0-9]+)_(.{1,})"


def get_sheet_info(text: str) -> dict:
    """
    It takes a string and returns a dictionary

    Args:
      text (str): str = the text that you want to search through

    Returns:
      A dictionary of dictionaries.
    """
    matches = re.finditer(PART_INFO_REGEX, text, re.MULTILINE)
    if len(re.findall(PART_INFO_REGEX, text, re.MULTILINE)) == 0:
        matches = re.finditer(OTHER_PART_INFO_REGEX, text, re.MULTILINE)
    return {
        match.group(1): {
            "Unit Number": match.group(2),
            "Part Name": match.group(3),
        }
        for match in matches
    }


def parse_pdf(pdf_path: str) -> None:
    """
    It takes a pdf file path as an argument, opens the pdf file, and then iterates through each page of
    the pdf file.

    For each page, it extracts the text from the page, and then passes the text to a function called
    get_sheet_info.

    The get_sheet_info function returns a dictionary of information about the page.

    The dictionary is then added to a global dictionary called file_info.

    The file_info dictionary is a dictionary of dictionaries.

    The keys of the file_info dictionary are the page numbers of the pdf file.

    The values of the file_info dictionary are the dictionaries returned by the get_sheet_info function.


    The get_sheet_info function is defined below.

    Args:
      pdf_path (str): The path to the PDF file.
    """
    pdf_file = fitz.open(pdf_path)
    pages = list(range(pdf_file.page_count))
    for page_number in range(pdf_file.page_count):
        print(f"\tParsing page {page_number + 1} of {pdf_file.page_count}")
        if page_number in pages:
            page = pdf_file[page_number]
            page_lines = page.get_text("text")
            with open("output.txt", "a") as f:
                f.write(page_lines)
            file_info[page_number] = get_sheet_info(page_lines)


def convert_pdf_to_images(pdf_path: str) -> None:
    """
    It takes a PDF file and converts it to a list of images

    Args:
      pdf_path (str): The path to the PDF file you want to convert.
    """
    pages = convert_from_path(pdf_path, 500)
    for i, page in enumerate(pages):
        print(
            f"\t[+] Saving page {i + 1} of {len(pages)} to {program_directory}/images/{i}.jpg"
        )
        page.save(f"{program_directory}/images/{i}.jpg", "JPEG")


def clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    It takes an image, applies a contrast limited adaptive histogram equalization, and returns the
    result

    Args:
      img: The input image.
      clip_limit: Contrast limit.
      grid_size: Size of grid for histogram equalization. Input image will be divided into equally sized
    rectangular tiles. This parameter defines the number of tiles in row and column.

    Returns:
      the image after applying the CLAHE algorithm.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def area_filter(min_area, input_image):
    """
    It takes an image and a minimum area, and returns a binary image where only the connected components
    with an area greater than or equal to the minimum area are set to 255

    Args:
      min_area: The minimum area of a component to be kept.
      input_image: The image to be filtered.

    Returns:
      a binary image where the remaining components are white and the rest is black.
    """
    (
        components_number,
        labeled_image,
        component_stats,
        component_centroids,
    ) = cv2.connectedComponentsWithStats(input_image, connectivity=4)

    remaining_component_labels = [
        i for i in range(1, components_number) if component_stats[i][4] >= min_area
    ]

    return np.where(
        np.isin(labeled_image, remaining_component_labels) == True, 255, 0
    ).astype("uint8")


def get_number(image, whitelist: str = "0123456789") -> str:
    """
    It takes an image and returns the number in the image

    Args:
      image: The image to be processed.
      whitelist (str): The characters that are allowed to be recognized. Defaults to 0123456789

    Returns:
      A string of the number that was found in the image.
    """
    # Use Tesseract to extract the number
    result = pytesseract.image_to_string(
        image,
        config=f"-l eng --psm 10 --oem 3 -c tessedit_char_whitelist={whitelist}"
    )

    return result.strip()

def resize_image(image, scale_percent: int):
    """
    It takes an image and a scale percentage as input, and returns a resized image

    Args:
      image: The image to be resized
      scale_percent (int): The percentage by which the image is to be scaled.

    Returns:
      The image is being resized to the dimensions specified by the scale_percent parameter.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def increase_canvas_size(image, canvas_size=(500, 500)):
    """
    Increase the canvas size of an image and center the original image within it.

    Args:
      image: The input image to be centered on a larger canvas.
      canvas_size: A tuple (width, height) defining the size of the new canvas.

    Returns:
      The new image with the original image centered on a larger canvas.
    """
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]
    
    # Create a blank canvas with the specified size and white background
    canvas = np.ones((canvas_size[1], canvas_size[0]), dtype=np.uint8) * 255
    
    # Calculate the top-left corner where the image should be placed to center it
    x_offset = (canvas_size[0] - original_width) // 2
    y_offset = (canvas_size[1] - original_height) // 2
    
    # Place the original image on the canvas
    canvas[y_offset:y_offset + original_height, x_offset:x_offset + original_width] = image
    
    return canvas


def embed_images() -> None:
    """
    It takes an image, finds the sheet, finds the circles, finds the numbers, and embeds the numbers
    into the image.
    """
    images = os.listdir(f"{program_directory}/images")
    images = natsorted(images)
    for i, image_path in enumerate(images):
        image = cv2.imread(f"{program_directory}/images/{image_path}", 1)
        possible_numbers = list(file_info[i].keys())
        lower = [140, 140, 140]
        upper = [180, 180, 180]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        ret, thresh = cv2.threshold(mask, 255, 255, 255)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            sheet_x, sheet_y, w, h = cv2.boundingRect(c)
            sheet_x -= 50
            sheet_y -= 50
            sheet = image[sheet_y : sheet_y + h + 100, sheet_x : sheet_x + w + 100]

            # Create a mask for the blue circles
            blue_lower = np.array([100, 0, 0], dtype="uint8")
            blue_upper = np.array([255, 50, 50], dtype="uint8")
            blue_mask = cv2.inRange(sheet, blue_lower, blue_upper)
            blurred_mask = cv2.GaussianBlur(blue_mask, (3, 3), 0)

            # Use the blue_mask as the binary image
            binary_image = blurred_mask.copy()

            binary_image = cv2.bitwise_not(binary_image)

            circles_contours = cv2.findContours(
                binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )[-2]
            # test_binary_image = resize_image(binary_image, 20)
            # cv2.imshow("binary_image", test_binary_image)
            # cv2.waitKey(0)
            # for i, circle in enumerate(circles_contours):
            #     print(i, cv2.contourArea(circle))

            min_area: int = 5000
            max_area: int = 10000
            circles = [
                circle
                for circle in circles_contours
                if min_area < cv2.contourArea(circle) < max_area
            ]
            unknown_circles = {}
            for circle in circles:
                x, y, w, h = cv2.boundingRect(circle)
                cropped = image[
                    sheet_y + y + 15 : sheet_y + y - 15 + h,
                    sheet_x + x + 15 : sheet_x + x + w - 15,
                ]
                
                # Convert cropped image to grayscale
                gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                
                # Create a binary mask to isolate dark areas (black)
                _, black_mask = cv2.threshold(gray_cropped, 50, 155, cv2.THRESH_BINARY_INV)
                
                # Apply the mask to the cropped image
                black_only = cv2.bitwise_and(cropped, cropped, mask=black_mask)

                # Convert the cropped image to grayscale to remove colors
                # Alternatively, just convert the masked output to grayscale
                # This will keep only the dark regions
                gray_black_only = cv2.cvtColor(black_only, cv2.COLOR_BGR2GRAY)

                # Optional: Threshold to keep only black areas
                _, final_mask = cv2.threshold(gray_black_only, 50, 255, cv2.THRESH_BINARY_INV)
                
                # Final output retains only black areas
                black_final = cv2.bitwise_and(black_only, black_only, mask=final_mask)

                if black_final is None or np.sum(black_final) == 0:
                    continue

                # Convert to grayscale
                gray_image = cv2.cvtColor(black_final, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Dilate the image to enhance the boldness
                kernel = np.ones((3, 3), np.uint8)
                dilated_image = cv2.dilate(thresh_image, kernel, iterations=1)
                dilated_image = resize_image(dilated_image, 200)
                dilated_image = increase_canvas_size(dilated_image, canvas_size=(800, 800))
                dilated_image = cv2.blur(dilated_image,(7, 7))
                text = (
                    get_number(dilated_image, whitelist="".join(possible_numbers))
                    .replace("\n", "")
                    .replace(" ", "")
                )
                try:
                    possible_numbers.pop(possible_numbers.index(text))
                except ValueError:
                    print("Could not find the number.")
                    unknown_circles = {text: circle}

                # print(cv2.contourArea(circle), text)
                # cv2.imshow("dilated_image", dilated_image)
                # cv2.waitKey(0)
                with contextlib.suppress(KeyError):
                    file_info[i][text].update({"image": [sheet_x + x, sheet_y + y]})
            if len(possible_numbers) == 1:
                file_info[i][possible_numbers[0]].update(unknown_circles)
                print("One number was not found, simply adding it.")
                # print(f"Numbers remaining: {possible_numbers}")
        print(f"\t[+] Proccessed {i + 1} of {len(images)}")
    for sheet_id in list(file_info.keys()):
        img = cv2.imread(f"{program_directory}/images/{sheet_id}.jpg")
        for part in list(file_info[sheet_id].keys()):
            try:
                x = file_info[sheet_id][part]["image"][0]
                y = file_info[sheet_id][part]["image"][1]
            except KeyError:
                continue
            cv2.putText(
                img,
                file_info[sheet_id][part]["Unit Number"],
                (x - 40, y + 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                5,
                cv2.LINE_AA,
            )
            with contextlib.suppress(Exception):
                cv2.putText(
                    img,
                    part_names[file_info[sheet_id][part]["Part Name"]],
                    (x + 80, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
        cv2.imwrite(f"{program_directory}/images/{sheet_id}.jpg", img)
        print(f"\t[+] Labeled {sheet_id+1} of {len(list(file_info.keys()))}")


def compile_back_to_pdf(pdf_file_name) -> None:
    """
    It takes all the images in the images folder and compiles them into a PDF file

    Args:
      pdf_file_name: The name of the PDF file you want to compile back to PDF.
    """
    images = os.listdir(f"{program_directory}/images")
    images = natsorted(images)
    new_pages = [Image.open(f"{program_directory}/images/{image}") for image in images]
    new_pages[0].save(
        pdf_file_name.replace(".pdf", " - Embeded.pdf"),
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=new_pages[1:],
    )


if __name__ == "__main__":
    file_name: str = sys.argv[-1].split("\\")[-1]
    directory_of_file: str = os.getcwd()
    pdf_file_name = f"{directory_of_file}/{file_name}"
    # pdf_file_name = r"C:\Users\Jared\Downloads\James Rachel.pdf"``
    # pdf_file_name = r"C:\Users\CarpenterShop\Documents\test1.pdf"
    # pdf_file_name = r"C:\Users\CarpenterShop\Documents\Pointer house 3.pdf"
    Path(f"{program_directory}/images").mkdir(parents=True, exist_ok=True)
    print(f"[ ] Getting {pdf_file_name} info...")
    parse_pdf(pdf_file_name)
    print("[+] Successfully got all PDF info.")
    print(f"[ ] Converting {pdf_file_name} to images...")
    convert_pdf_to_images(pdf_file_name)
    print(f"[+] Successfully converted {pdf_file_name} to images.")
    print(f"[ ] Labeling {pdf_file_name}.")
    embed_images()
    print(f"[+] Successfully Labeled {pdf_file_name}.")
    print(
        f"[ ] Compiling {pdf_file_name} as {pdf_file_name.replace('.pdf', ' - Embeded.pdf')}."
    )
    compile_back_to_pdf(pdf_file_name)
    print(
        f"[+] Successfully compiled {pdf_file_name.replace('.pdf', ' - Embeded.pdf')}."
    )
    shutil.rmtree(f"{program_directory}/images")
    # os.remove(pdf_file_name)
    print("Finished!")
    webbrowser.open_new(f"file:\\{pdf_file_name.replace('.pdf', ' - Embeded.pdf')}")
