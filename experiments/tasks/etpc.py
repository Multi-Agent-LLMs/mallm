# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""ETPC: The Extended Typology Paraphrase Corpus"""

import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import datasets
import numpy as np
from datasets.tasks import TextClassification
from lxml import etree

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{kovatchev-etal-2018-etpc,
    title = "{ETPC} - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negation",
    author = "Kovatchev, Venelin  and
      Mart{\'\i}, M. Ant{\`o}nia  and
      Salam{\'o}, Maria",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1221",
}
"""

_DESCRIPTION = """\
The EPT typology addresses several practical limitations of existing paraphrase typologies: it is the first typology that copes with the non-paraphrase pairs in the paraphrase identification corpora and distinguishes between contextual and habitual paraphrase types. ETPC is the largest corpus to date annotated with atomic paraphrase types.
"""

_HOMEPAGE = "https://github.com/venelink/ETPC"

_LICENSE = "Unknown"

_URLS = [
    "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/text_pairs.xml",
    "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/textual_paraphrases.xml",
]


class ETPC(datasets.GeneratorBasedBuilder):
    """ETPC dataset."""

    VERSION = datasets.Version("0.95.0")

    def _info(self):
        features = datasets.Features(
            {
                "idx": datasets.Value("string"),
                "sentence1": datasets.Value("string"),
                "sentence2": datasets.Value("string"),
                "sentence1_tokenized": datasets.Sequence(
                    datasets.Value("string")
                ),
                "sentence2_tokenized": datasets.Sequence(
                    datasets.Value("string")
                ),
                "etpc_label": datasets.Value("int8"),
                "mrpc_label": datasets.Value("int8"),
                "negation": datasets.Value("int8"),
                "paraphrase_types": datasets.Sequence(
                    datasets.Value("string")
                ),
                "paraphrase_type_ids": datasets.Sequence(
                    datasets.Value("string")
                ),
                "sentence1_segment_location": datasets.Sequence(
                    datasets.Value("int32")
                ),
                "sentence2_segment_location": datasets.Sequence(
                    datasets.Value("int32")
                ),
                "sentence1_segment_location_indices": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"))
                ),
                "sentence2_segment_location_indices": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"))
                ),
                "sentence1_segment_text": datasets.Sequence(
                    datasets.Value("string")
                ),
                "sentence2_segment_text": datasets.Sequence(
                    datasets.Value("string")
                ),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file_paths": dl_manager.iter_files(dl_dir),
                },
            ),
        ]

    def _generate_examples(self, file_paths):
        file_paths = list(file_paths)
        text_pairs_path = file_paths[0]
        paraphrase_types_path = file_paths[1]

        parser = etree.XMLParser(encoding="utf-8", recover=True)

        tree_text_pairs = etree.parse(text_pairs_path, parser=parser)
        tree_paraphrase_types = etree.parse(
            paraphrase_types_path, parser=parser
        )

        root_text_pairs = tree_text_pairs.getroot()
        root_paraphrase_types = tree_paraphrase_types.getroot()

        idx = 0

        for row in root_text_pairs:
            current_pair_id = row.find(".//pair_id").text
            paraphrase_types = root_paraphrase_types.xpath(
                f".//pair_id[text()='{current_pair_id}']/parent::relation/type_name/text()"
            )
            paraphrase_type_ids = root_paraphrase_types.xpath(
                f".//pair_id[text()='{current_pair_id}']/parent::relation/type_id/text()"
            )
            sentence1_segment_location = root_paraphrase_types.xpath(
                f".//pair_id[text()='{current_pair_id}']/parent::relation/s1_scope/text()"
            )
            sentence2_segment_location = root_paraphrase_types.xpath(
                f".//pair_id[text()='{current_pair_id}']/parent::relation/s2_scope/text()"
            )
            sentence1_segment_text = root_paraphrase_types.xpath(
                f".//pair_id[text()='{current_pair_id}']/parent::relation/s1_text/text()"
            )
            sentence2_segment_text = root_paraphrase_types.xpath(
                f".//pair_id[text()='{current_pair_id}']/parent::relation/s2_text/text()"
            )

            sentence1_tokenized = row.find(".//sent1_tokenized").text.split(
                " "
            )
            sentence2_tokenized = row.find(".//sent2_tokenized").text.split(
                " "
            )

            sentence1_segment_location_full = np.zeros(
                len(sentence1_tokenized)
            )
            sentence2_segment_location_full = np.zeros(
                len(sentence2_tokenized)
            )

            sentence1_segment_indices = []
            sentence2_segment_indices = []

            for (
                sentence1_segment_locations,
                sentence2_segment_locations,
                paraphrase_type_id,
            ) in zip(
                sentence1_segment_location,
                sentence2_segment_location,
                paraphrase_type_ids,
            ):
                segment_locations_1 = [
                    int(i) for i in sentence1_segment_locations.split(",")
                ]
                sentence1_segment_indices.append(segment_locations_1)
                sentence1_segment_location_full[segment_locations_1] = [
                    paraphrase_type_id
                ] * len(segment_locations_1)

                segment_locations_2 = [
                    int(i) for i in sentence2_segment_locations.split(",")
                ]
                sentence2_segment_indices.append(segment_locations_2)
                sentence2_segment_location_full[segment_locations_2] = [
                    paraphrase_type_id
                ] * len(segment_locations_2)

            yield idx, {
                "idx": row.find(".//pair_id").text + "_" + str(idx),
                "sentence1": row.find(".//sent1_raw").text,
                "sentence2": row.find(".//sent2_raw").text,
                "sentence1_tokenized": sentence1_tokenized,
                "sentence2_tokenized": sentence2_tokenized,
                "etpc_label": int(row.find(".//etpc_label").text),
                "mrpc_label": int(row.find(".//mrpc_label").text),
                "negation": int(row.find(".//negation").text),
                "paraphrase_types": paraphrase_types,
                "paraphrase_type_ids": paraphrase_type_ids,
                "sentence1_segment_location": sentence1_segment_location_full,
                "sentence2_segment_location": sentence2_segment_location_full,
                "sentence1_segment_location_indices": sentence1_segment_indices,
                "sentence2_segment_location_indices": sentence2_segment_indices,
                "sentence1_segment_text": sentence1_segment_text,
                "sentence2_segment_text": sentence2_segment_text,
            }
            idx += 1
