from typing import List, Optional, Tuple
from .model_skeletons import LinkedMention, MentionsOfDoc, Mention, NER, NED


import spacy


class SpacyNER(NER):
    def __init__(
        self, pipeline_name: str = "en_core_web_sm", prefer_gpu: bool = False
    ) -> None:
        if prefer_gpu:
            spacy.prefer_gpu()
        self.model = spacy.load(pipeline_name)

    def recognize_entities_in_docs(self, text: str) -> list[dict]:
        output = self.model.pipe(text)
        rearranged_output = []
        for doc in output:
            rearranged_output.append(
                MentionsOfDoc(
                    text=doc.text,
                    entities=[
                        Mention(
                            text=ent.text,
                            start=ent.start_char,
                            end=ent.end_char,
                            label=ent.label_,
                        )
                        for ent in doc.ents
                    ],
                )
            )
        return rearranged_output


from refined.data_types.base_types import Span
from refined.inference.processor import Refined
from dataclasses import asdict


class ReFiNED(NED):
    def __init__(
        self,
        ned_model_path="wikipedia_model",
        entity_set="wikipedia",
        no_cuda=False,
    ) -> None:

        self.model = Refined.from_pretrained(
            model_name=ned_model_path, entity_set=entity_set
        )

    @staticmethod
    def model_input_formatting(
        mention_surfaceform: str, context: str, other_info: dict = None
    ):
        return (
            Span(
                text=mention_surfaceform,
                start=other_info.start,
                ln=other_info.end - other_info.start,
            ),
            context,
        )

    def model_output_formatting(
        self, mention_ned_result, mention_ner_result=None
    ) -> dict:
        mention_as_dict = asdict(mention_ned_result)

        def get_entity_id(mention_as_dict):
            if (
                not mention_as_dict["predicted_entity"]
                or not mention_as_dict["predicted_entity"]["wikidata_entity_id"]
            ):
                if mention_as_dict["candidate_entities"]:
                    return (
                        mention_as_dict["candidate_entities"][0][0],
                        mention_as_dict["text"],
                    )
                else:
                    return None, None
            else:
                return (
                    mention_as_dict["predicted_entity"]["wikidata_entity_id"],
                    mention_as_dict["predicted_entity"]["wikipedia_entity_title"],
                )

        wiki_id, wiki_title = get_entity_id(mention_as_dict)
        e = LinkedMention(
            id=wiki_id,
            title=wiki_title,
            mention=mention_as_dict["text"],
            label=mention_ner_result.label,
        )

        return e

    def disambiguate_mentions_in_docs(self, mentions_batch: list):
        # refined does in-place modifications, deep copy the Spans if you need
        spanss = [[mention[0] for mention in mentions_batch]]
        texts = [mention[1] for mention in mentions_batch]
        predictions = self.model.process_text_batch(texts=texts, spanss=spanss)
        return spanss[0]


import blink.main_dense as main_dense
import argparse


class BlinkNED(NED):
    """
    Named Entity Disambiguation
    """

    def __init__(
        self,
        model_path="/home/bodor/models/BLINK/models/",
        no_cuda=False,
    ) -> None:
        self.config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 1,
            "biencoder_model": model_path + "biencoder_wiki_large.bin",
            "biencoder_config": model_path + "biencoder_wiki_large.json",
            "entity_catalogue": model_path + "entity.jsonl",
            "entity_encoding": model_path + "all_entities_large.t7",
            "crossencoder_model": model_path + "crossencoder_wiki_large.bin",
            "crossencoder_config": model_path + "crossencoder_wiki_large.json",
            "fast": False,  # set this to be true if speed is a concern
            "output_path": "logs/",  # logging directory
        }
        self.args = argparse.Namespace(**self.config)
        self.models = main_dense.load_models(self.args, logger=None)
        self.title2id = self.models[5]
        self.title2id

    @staticmethod
    def model_input_formatting(mention_surfaceform, context, other_info):
        return {
            "mention": mention_surfaceform,
            "context_left": context[: other_info.start],
            "context_right": context[other_info.end :],
            "label": "unknown",
            "label_id": -1,
        }

    def model_output_formatting(self, mention_ned_result, mention_ner_result):
        e = LinkedMention(
            id=self.title2id[mention_ned_result[0]],
            title=mention_ned_result[0],
            mention=mention_ner_result.text,
            label=mention_ner_result.label,
        )
        return e

    def disambiguate_mentions_in_docs(self, mentions_batch):
        (
            _,
            _,
            _,
            _,
            _,
            predictions,
            scores,
        ) = main_dense.run(self.args, None, *self.models, test_data=mentions_batch)
        return predictions
