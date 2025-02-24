from typing import List, Dict
from dataclasses import dataclass


@dataclass
class LinkedMention:
    """
    This class represents mention linked to an ID of named entity.
    """

    id: str
    title: str
    mention: str
    label: str


@dataclass
class Mention:
    """
    This class represents detected mention.
    """

    text: str
    start: int
    end: int
    label: str


@dataclass
class MentionsOfDoc:
    """
    This class represents doc with all detected mentions within it.
    """

    text: str
    entities: List[Mention]


class NER:
    """
    Named Entity Recognition
    """

    def __init__(
        self, pipeline_name: str = "en_core_web_sm", prefer_gpu: bool = True
    ) -> None:
        pass

    def recognize_entities_in_docs(self, text: str) -> List[MentionsOfDoc]:
        pass


class NED:
    """
    Named Entity Disambiguation
    """

    def __init__(
        self,
        model_path: str = "",
        no_cuda: bool = False,
    ) -> None:
        pass

    @staticmethod
    def model_input_formatting(
        mention_surfaceform: str, context: str, other_info: Dict = None
    ):
        pass

    def model_output_formatting(
        self, mention_ned_result, mention_ner_result
    ) -> LinkedMention:
        pass

    def disambiguate_mentions_in_docs(self, mentions_batch: List):
        pass
