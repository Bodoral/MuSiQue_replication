from .model_skeletons import NER, NED
from .supported_ner_ned import SpacyNER, ReFiNED, BlinkNED


class NEL:
    """
    Named Entity Linking i.e. NER and NED
    """

    def __init__(
        self,
        ner_model: NER = SpacyNER(),
        ned_model: NED = BlinkNED(),
        no_cuda=False,
    ) -> None:
        self.ner = ner_model
        self.ned = ned_model
        self.no_cuda = no_cuda

    def __clean_mention(self, mention: str) -> str:
        clean_mention = mention.strip()
        clean_mention = " ".join(clean_mention.split())
        return clean_mention

    def link_entities_in_docs(self, docs: list) -> list[dict]:
        docs_mentions = self.ner.recognize_entities_in_docs(docs)

        splits = []
        mentions_batch = []
        for doc_mentions in docs_mentions:
            context = doc_mentions.text
            splits.append(len(doc_mentions.entities))
            for mention in doc_mentions.entities:
                mention_text = mention.text
                mention_text = self.__clean_mention(mention_text)
                mentions_batch.append(
                    self.ned.model_input_formatting(mention_text, context, mention)
                )

        if mentions_batch:
            search_results = self.ned.disambiguate_mentions_in_docs(mentions_batch)
        else:
            search_results = []

        start = 0
        rearranged_results = []
        for split in splits:
            rearranged_results.append(search_results[start : start + split])
            start += split

        if mentions_batch:
            for doc_index in range(len(rearranged_results)):
                for entity_index in range(len(rearranged_results[doc_index])):
                    e = self.ned.model_output_formatting(
                        mention_ned_result=rearranged_results[doc_index][entity_index],
                        mention_ner_result=docs_mentions[doc_index].entities[
                            entity_index
                        ],
                    )
                    rearranged_results[doc_index][entity_index] = e

        return rearranged_results


if __name__ == "__main__":
    docs = [
        """The Nakba is the ethnic cleansing of Palestinian through their violent displacement and dispossession of land, property, and belongings before and during the 1948 Arab-Israeli war that followed Israel’s establishment.""",
        """Zionist military forces expelled at least 750,000 Palestinians from their homes and lands and captured 78 percent of Palestine. The remaining 22 percent was divided into what are now the occupied West Bank and the besieged Gaza Strip."""
        """On April 9, 1948, Zionist forces committed one of the most infamous massacres of the war in the village of Deir Yassin on the western outskirts of Jerusalem. More than 110 men, women and children were killed by members of the pre-Israeli-state Irgun and Stern Gang Zionist militias.""",
    ]

    nel = NEL()
    rearranged_results = nel.link_entities_in_docs(docs)
    print(rearranged_results)
