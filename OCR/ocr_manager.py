#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
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

from basic.generic_training_manager import GenericTrainingManager
import os
from PIL import Image
import pickle


class OCRManager(GenericTrainingManager):
    def __init__(self, params):
        super(OCRManager, self).__init__(params)
        if self.dataset is not None:
            self.params["model_params"]["vocab_size"] = len(self.dataset.charset)

    def generate_syn_line_dataset(self, name):
        """
        Generate synthetic line dataset from currently loaded dataset
        """
        dataset_name = list(self.params['dataset_params']["datasets"].keys())[0]
        path = os.path.join(os.path.dirname(self.params['dataset_params']["datasets"][dataset_name]), name)
        os.makedirs(path, exist_ok=True)
        charset = set()
        dataset = None
        gt = {
            "train": dict(),
            "valid": dict(),
            "test": dict()
        }
        for set_name in ["train", "valid", "test"]:
            set_path = os.path.join(path, set_name)
            os.makedirs(set_path, exist_ok=True)
            if set_name == "train":
                dataset = self.dataset.train_dataset
            elif set_name == "valid":
                dataset = self.dataset.valid_datasets["{}-valid".format(dataset_name)]
            elif set_name == "test":
                self.dataset.generate_test_loader("{}-test".format(dataset_name), [(dataset_name, "test"), ])
                dataset = self.dataset.test_datasets["{}-test".format(dataset_name)]

            samples = list()
            index = 0
            handwritten = self.params["dataset_params"].get("config", {}).get("synthetic_data", {}).get("config", {}).get("not_synthetic", False)
            while index < len(dataset.samples):
                #for sample in dataset.samples:
                sample = dataset.__getitem__(index, raw=handwritten)
                if handwritten:
                    for one_line in sample["lines"]:
                        charset = charset.union(set(one_line["text"]))
                        samples.append({
                            "path": sample["path"],
                            "label": one_line["text"],
                            "top": one_line["top"],
                            "bottom": one_line["bottom"],
                            "left": one_line["left"],
                            "right": one_line["right"],
                            "nb_cols": 1,
                            "index": index,
                        })
                        '''
                        if self.params["dataset_params"].get("config", {}).get("synthetic_data", {}).get("config", {}).get("include_word_images", False):
                            for one_word in one_line["words"]:
                                charset = charset.union(set(one_word["text"]))
                                samples.append({
                                    "path": sample["path"],
                                    "label": one_word["text"],
                                    "top": one_word["top"],
                                    "bottom": one_word["bottom"],
                                    "left": one_word["left"],
                                    "right": one_word["right"],
                                    "nb_cols": 1,
                                    "index": index,
                                })
                        '''
                else:
                    for line_label in sample["label"].split("\n"):
                        for chunk in [line_label[i:i+100] for i in range(0, len(line_label), 100)]:
                            charset = charset.union(set(chunk))
                            if len(chunk) > 0:
                                samples.append({
                                    "path": sample["path"],
                                    "label": chunk,
                                    "nb_cols": 1,
                                })
                index += 1

            for i, sample in enumerate(samples):
                ext = sample['path'].split(".")[-1]
                img_name = "{}_{}.{}".format(set_name, i, ext)
                img_path = os.path.join(set_path, img_name)
                print(os.path.abspath(img_path))

                if self.params["dataset_params"].get("config", {}).get("synthetic_data", {}).get("config", {}).get("not_synthetic", False):
                    img = dataset.generate_real_line_image(sample)
                else:
                    img = dataset.generate_typed_text_line_image(sample["label"])
                Image.fromarray(img).save(img_path)
                gt[set_name][img_name] = {
                    "text": sample["label"],
                    "nb_cols": sample["nb_cols"] if "nb_cols" in sample else 1
                }
                if "top" in sample:
                    gt[set_name][img_name]["top"] = sample["top"]
                    gt[set_name][img_name]["left"] = sample["left"]
                    gt[set_name][img_name]["bottom"] = sample["bottom"]
                    gt[set_name][img_name]["right"] = sample["right"]
                    gt[set_name][img_name]["index"] = sample["index"]

        with open(os.path.join(path, "labels.pkl"), "wb") as f:
            pickle.dump({
                "ground_truth": gt,
                "charset": sorted(list(charset)),
            }, f)
