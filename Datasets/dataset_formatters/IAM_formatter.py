#  Copyright Open Universiteit Nederland 2025
#  contributors :
#  - Michel Vos
#
#
#  This software is a computer program written in Python  whose purpose is to
#  provide public implementation of deep learning works, in pytorch.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.


#from Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
import os
import sys
from os.path import dirname
DOSSIER_COURRANT = dirname(os.path.abspath(__file__))
ROOT_FOLDER = dirname(dirname(dirname(DOSSIER_COURRANT)))
sys.path.append(ROOT_FOLDER)


from Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

# Layout begin-token to end-token
SEM_MATCHING_TOKENS = {
            "ⓑ": "Ⓑ",  # paragraph (body)
            "ⓟ": "Ⓟ",  # page
            "ⓢ": "Ⓢ",  # section (=linked annotation + body)
        }


class IAMDatasetFormatter(OCRDatasetFormatter):
    def __init__(self, level, set_names=["train", "valid", "test"], dpi=150, end_token=True, sem_token=True):
        super(IAMDatasetFormatter, self).__init__("IAM", level, "_sem" if sem_token else "", set_names)

        self.map_datasets_files.update({
            "IAM": {
                "page": {
                    "arx_files": ["IAM_train.gz", "IAM_validate.gz", "IAM_test.gz"],
                    "needed_files": [],
                    "format_function": self.format_IAM_page,
                },
            }
        })
        self.dpi = dpi
        self.end_token = end_token
        self.sem_token = sem_token
        self.matching_token = SEM_MATCHING_TOKENS

    def init_format(self):
        super().init_format()
        
        os.rename(os.path.join(self.temp_fold, "validate"), os.path.join(self.temp_fold, "valid"))
        for set_name in ["train", "valid", "test"]:
            for filename in os.listdir(os.path.join(self.temp_fold, set_name, "forms")):
                filepath = os.path.join(self.temp_fold, set_name, "forms", filename)
                if os.path.isfile(filepath):
                    os.rename(filepath, os.path.join(self.temp_fold, set_name, filename))
            os.rmdir(os.path.join(self.temp_fold, set_name, "forms"))


    def preformat_IAM(self):
        """
        Extract all information from IAM dataset and correct some mistakes
        """
        def get_word_bbox(word_element):
            x_min = float("inf")
            y_min = float("inf")
            x_max = -float("inf")
            y_max = -float("inf")

            for cmp in word_element.findall("./cmp"):
                x = int(cmp.get("x"))
                y = int(cmp.get("y"))
                width = int(cmp.get("width"))
                height = int(cmp.get("height"))

                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + width)
                y_max = max(y_max, y + height)

            return x_min, y_min, x_max, y_max

        def get_line_bbox(line_element):
            x_min = float("inf")
            y_min = float("inf")
            x_max = -float("inf")
            y_max = -float("inf")

            for word in line_element.findall("./word"):
                wx1, wy1, wx2, wy2 = get_word_bbox(word)
                x_min = min(x_min, wx1)
                y_min = min(y_min, wy1)
                x_max = max(x_max, wx2)
                y_max = max(y_max, wy2)

            return x_min, y_min, x_max, y_max

        def get_text_region_bbox(text_region_element):
            x_min = float("inf")
            y_min = float("inf")
            x_max = -float("inf")
            y_max = -float("inf")

            for line in text_region_element.findall("./line"):
                lx1, ly1, lx2, ly2 = get_line_bbox(line)
                x_min = min(x_min, lx1)
                y_min = min(y_min, ly1)
                x_max = max(x_max, lx2)
                y_max = max(y_max, ly2)

            return x_min, y_min, x_max, y_max   


        def box_to_points(x_min, y_min, x_max, y_max):
            """
            Convert bounding box coordinates to points dictionary
            """
            return {
                "left": x_min,
                "top":  y_min,
                "right": x_max,
                "bottom": y_max
            }

        dataset = {
            "train": list(),
            "valid": list(),
            "test": list(),
        }
        for set_name in ["train", "valid", "test"]:
            img_fold_path = os.path.join(self.temp_fold, set_name)
            xml_fold_path = os.path.join(self.temp_fold, set_name, "xml")
            for xml_file_name in sorted(os.listdir(xml_fold_path)):
                if xml_file_name.split(".")[-1] != "xml":
                    continue
                filename = xml_file_name.split(".")[0]
                img_path = os.path.join(img_fold_path, filename + ".png")
                xml_file_path = os.path.join(xml_fold_path, xml_file_name)
                xml_root = ET.parse(xml_file_path).getroot()

                with Image.open(img_path) as img:
                    width, height = img.size

                page_dict = {
                    "label": list(),
                    "text_regions": list(),
                    "img_path": img_path,
                    "width": width,
                    "height": height
                }
                text_regions = xml_root.findall("handwritten-part")
                for text_region in text_regions:
                    text_region_dict = {
                        "label": list(),
                        "lines": list(),
                        "coords": box_to_points(*get_text_region_bbox(text_region))
                    }
                    text_lines = text_region.findall("line")
                    for text_line in text_lines:
                        text_line_label = text_line.get("text")
                        if text_line_label is None:
                            print("ignored null line{}".format(page_dict["img_path"]))
                            continue
                        label = self.format_text_label(text_line_label)
                        text_line_dict = {
                            "label": label,
                            "coords": box_to_points(*get_line_bbox(text_line)),
                            "baseline_coords": box_to_points(*get_line_bbox(text_line))
                        }
                        text_line_dict["label"] = text_line_dict["label"]
                        text_line_dict["words"] = list()
                        text_region_dict["label"].append(text_line_dict["label"])
                        text_region_dict["lines"].append(text_line_dict)
                        words = text_line.findall("word")
                        for word in words:
                            word_text = word.get("text")
                            if word_text is None:
                                print("ignored null word {}".format(page_dict["img_path"]))
                                continue
                            word_label = self.format_text_label(word_text)
                            word_box = box_to_points(*get_word_bbox(word))
                            word_data = {
                                "label": word_label,
                                "coords": word_box,
                            }
                            text_line_dict["words"].append(word_data)
                    if text_region_dict["label"] == list():
                        print("ignored null region {}".format(page_dict["img_path"]))
                        continue
                    text_region_dict["label"] = self.format_text_label("\n".join(text_region_dict["label"]))
                    text_region_dict["baseline_coords"] = {
                        "left": min([line["baseline_coords"]["left"] for line in text_region_dict["lines"]]),
                        "right": max([line["baseline_coords"]["right"] for line in text_region_dict["lines"]]),
                        "bottom": max([line["baseline_coords"]["bottom"] for line in text_region_dict["lines"]]),
                        "top": min([line["baseline_coords"]["top"] for line in text_region_dict["lines"]]),
                    }
                    page_dict["label"].append(text_region_dict["label"])
                    page_dict["text_regions"].append(text_region_dict)
                    page_dict["label"] = self.format_text_label("\n".join(page_dict["label"]))
                    dataset[set_name].append(page_dict)

        return dataset
    
    def format_IAM_page(self):
        """
        Format the IAM dataset at single-page level
        """
        def get_brightest_color(img, left, top, right, bottom):
            box = (left+1, top+1, right-1, bottom-1)
            region = img.crop(box)
            arr = np.array(region)
            if img.mode == "L":
                # Grayscale: max value
                bright_pixels = arr[arr > 128]
                if bright_pixels.size > 0:
                    return int(bright_pixels.mean())
                else:
                    return int(arr.max())  
            else:
                # RGB: find pixel with max sum (brightness)
                flat = arr.reshape(-1, arr.shape[-1])
                idx = np.argmax(flat.sum(axis=1))
                return tuple(flat[idx])
        """
        Remove the printed texts
        """
        def blank(img, left, top, right, bottom):
            draw = ImageDraw.Draw(img)
            width, height = img.size
            fill_color = get_brightest_color(img, left, top, right, bottom)
            left = max(0, left - 2)
            right= min(width, right + 2)
            top = max(0, top - 2)
            bottom = min(height, bottom + 1)

            #if img.mode == "L":
            #    fill_color = 255  # white for grayscale
            #else:
            #    fill_color = fill  
            # top
            draw.rectangle([0,0,width,top], fill=fill_color)
            # bottom
            draw.rectangle([0, bottom, width, height], fill=fill_color)
            # Left
            draw.rectangle([0, top, left, bottom], fill=fill_color)
            # Right
            draw.rectangle([right, top, width, bottom], fill=fill_color)
            return img
        
        def load_resize_save_remove(page, target_path, source_dpi, target_dpi):
            """
            Load image, apply resolution modification and save it
            """
            source_path=page["img_path"]
            if source_dpi != target_dpi:
                img = Image.open(source_path)
                coords=page["text_regions"][0]["coords"]
                img = blank(img, coords["left"], coords["top"], coords["right"], coords["bottom"])
                img = self.resize(img, source_dpi, target_dpi)
                img = Image.fromarray(img)
                img.save(target_path)
            else:
                img = Image.open(source_path)
                coords=page["text_regions"][0]["coords"]
                img = blank(img, coords["left"], coords["top"], coords["right"], coords["bottom"])
                img.save(target_path)

        dataset = self.preformat_IAM()
        for set_name in ["train", "valid", "test"]:

            for i, page in enumerate(dataset[set_name]):
                new_img_name = "{}_{}.jpeg".format(set_name, i)
                new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)
                load_resize_save_remove(page, new_img_path, 300, self.dpi)
                new_label, sorted_text_regions, nb_cols, side = self.sort_text_regions(page["text_regions"], page["width"])
                paragraphs = list()
                for paragraph in page["text_regions"]:
                    paragraph_label = {
                        "label": paragraph["label"],
                        "lines": list(),
                        "mode": paragraph["mode"]
                    }
                    for line in paragraph["lines"]:
                            
                        line_data = {
                            "text": line["label"],
                            "top": line["coords"]["top"],
                            "bottom": line["coords"]["bottom"],
                            "left": line["coords"]["left"],
                            "right": line["coords"]["right"],
                            "words": list(),
                        }
                        for word in line["words"]:
                            word_box = self.adjust_coord_ratio(word["coords"], self.dpi / 300)
                            #line["label"] = line["label"].replace(word["label"], word["label"])
                            line_data["words"].append({
                                "text": word["label"],
                                "top": word_box["top"],
                                "bottom": word_box["bottom"],
                                "left": word_box["left"],
                                "right": word_box["right"],
                                })
                        paragraph_label["lines"].append(line_data)
                        #paragraph_label["lines"][-1] = self.adjust_coord_ratio(paragraph_label["lines"][-1], self.dpi / 300)
                        line_data = self.adjust_coord_ratio(line_data, self.dpi / 300)
                    paragraph_label["top"] = min([line["top"] for line in paragraph_label["lines"]])
                    paragraph_label["bottom"] = max([line["bottom"] for line in paragraph_label["lines"]])
                    paragraph_label["left"] = min([line["left"] for line in paragraph_label["lines"]])
                    paragraph_label["right"] = max([line["right"] for line in paragraph_label["lines"]])
                    paragraphs.append(paragraph_label)

                if self.sem_token:
                    if self.end_token:
                        new_label = "ⓟ" + new_label + "Ⓟ"
                    else:
                        new_label = "ⓟ" + new_label

                page_label = {
                    "text": new_label,
                    "paragraphs": paragraphs,
                    "nb_cols": nb_cols,
                    "side": side,
                    "top": min([pg["top"] for pg in paragraphs]),
                    "bottom": max([pg["bottom"] for pg in paragraphs]),
                    "left": min([pg["left"] for pg in paragraphs]),
                    "right": max([pg["right"] for pg in paragraphs]),
                    "page_width": int(np.array(Image.open(page["img_path"])).shape[1] * self.dpi / 300),
                    "lines": paragraph_label["lines"],
                }

                self.gt[set_name][new_img_name] = {
                    "text": new_label,
                    "nb_cols": nb_cols,
                    "pages": [page_label, ],
                }
                self.charset = self.charset.union(set(page["label"]))
        self.add_tokens_in_charset()


    def add_tokens_in_charset(self):
        """
        Add layout tokens to the charset
            "ⓑ": "Ⓑ",  # paragraph (body)
            "ⓟ": "Ⓟ",  # page
            "ⓢ": "Ⓢ",  # section (=linked annotation + body)

        """
        if self.sem_token:
            if self.end_token:
                self.charset = self.charset.union(set("ⓑⒷⓟⓅⓢⓈ"))
            else:
                self.charset = self.charset.union(set("ⓑⓟⓢ"))
    
    def update_label(self, label, start_token):
        """
        Add layout token to text region transcription
        """
        if self.sem_token:
            if self.end_token:
                return start_token + label + self.matching_token[start_token]
            else:
                return start_token + label
        return label

    def merge_group_tr(self, group, text_region):
        group["text_regions"].append(text_region)
        group["coords"]["top"] = min([tr["coords"]["top"] for tr in group["text_regions"]])
        group["coords"]["bottom"] = max([tr["coords"]["bottom"] for tr in group["text_regions"]])
        group["coords"]["left"] = min([tr["coords"]["left"] for tr in group["text_regions"]])
        group["coords"]["right"] = max([tr["coords"]["right"] for tr in group["text_regions"]])
        group["baseline_coords"]["top"] = min([tr["baseline_coords"]["top"] for tr in group["text_regions"]])
        group["baseline_coords"]["bottom"] = max([tr["baseline_coords"]["bottom"] for tr in group["text_regions"]])
        group["baseline_coords"]["left"] = min([tr["baseline_coords"]["left"] for tr in group["text_regions"]])
        group["baseline_coords"]["right"] = max([tr["baseline_coords"]["right"] for tr in group["text_regions"]])

    def is_annotation_alone(self, groups, page_width):
        for group in groups:
            if all([tr["coords"]["right"] < page_width / 2 and not tr["label"].replace(".", "").replace(" ", "").isdigit() for tr in group["text_regions"]]):
                return True
        return False

    def sort_text_regions(self, text_regions, page_width):
        """
        Establish reading order based on paragraph pixel position:
        page number then section by section: first all annotations, then associated body
        """
        nb_cols = 1
        groups = list()
        side = None
        for text_region in text_regions:
            groups.append({
                "coords": text_region["coords"].copy(),
                "baseline_coords": text_region["baseline_coords"].copy(),
                "text_regions": [text_region, ]
            })
        ordered_groups = sorted(groups, key=lambda g: g["coords"]["top"])
        sorted_text_regions = list()
        for group in ordered_groups:
            text_regions = group["text_regions"]
            left = [tr for tr in group["text_regions"] if tr["coords"]["right"] < page_width / 2]
            right = [tr for tr in group["text_regions"] if tr["coords"]["right"] >= page_width / 2]
            nb_cols = max(2 if len(left) > 0 else 1, nb_cols)
            for i, text_region in enumerate(sorted(right, key=lambda tr: tr["coords"]["top"])):
                sorted_text_regions.append(text_region)
                sorted_text_regions[-1]["mode"] = "body"
                sorted_text_regions[-1]["label"] = self.update_label(sorted_text_regions[-1]["label"], "ⓑ")
                if i == 0 and self.sem_token and len(left) == 0:
                    sorted_text_regions[-1]["label"] = "ⓢ" + sorted_text_regions[-1]["label"]
                if i == len(right)-1 and self.sem_token and self.end_token:
                    sorted_text_regions[-1]["label"] = sorted_text_regions[-1]["label"] + self.matching_token["ⓢ"]

        sep = "" if self.sem_token else "\n"
        new_label = sep.join(t["label"] for t in sorted_text_regions)
        return new_label, sorted_text_regions, nb_cols, side


if __name__ == "__main__":
    IAMDatasetFormatter("page", sem_token=True).format()
    IAMDatasetFormatter("page", sem_token=False).format()
