import spacy
from spacy.pipeline.ner import DEFAULT_NER_MODEL

class NER:
    def __init__(self,pipeline_name:str = "en_core_web_sm", prefer_gpu:bool = False) -> None:
        if prefer_gpu :
            spacy.prefer_gpu()
        self.nlp = spacy.load(pipeline_name)


    
    def get_docs_nemed_entities(self, text: list[str]):
        output = self.nlp.pipe(text)
        rearranged_output = []
        for doc in output:
            rearranged_output.append({"text":doc.text, 
                                    "entities":[{"text":ent.text, 
                                                "start":ent.start_char, 
                                                "end":ent.end_char,
                                                "label" :  ent.label_} for ent in doc.ents ]}

        )
        return rearranged_output


class EL:
    def __init__(self,) -> None:
      pass
      #Todo

    def link_multi_docs_entities(self, text: list[str]):
      pass
      #Todo:
