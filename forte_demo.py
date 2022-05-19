import pdb

import forte
import pprint
from forte.data.readers import TerminalReader
from fortex.nltk.nltk_processors import NLTKLemmatizer, NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter
from fortex.allennlp.allennlp_processors import AllenNLPProcessor
from ft.onto.base_ontology import Token, Sentence, PredicateLink
from forte.data.ontology.top import Query
from forte.data.caster import MultiPackBoxer
from es_query_generator import ElasticSearchQueryCreator
from fortex.elastic import ElasticSearchProcessor
from transformers import pipeline
import torch

"""
This code combines various steps described in the below-referenced forte tutorial into a single Question-Answering class.

    1. https://medium.com/casl-project/building-a-question-answering-system-part-1-query-understanding-in-18-lines-916110f9f2b2
    2. https://medium.com/casl-project/building-a-question-answering-system-part-2-document-retrieval-a84f57655d7e  
    3. https://medium.com/casl-project/building-a-question-answering-system-part-3-answer-extraction-5b9a4bd31e18

Intended for reference -- actually running this code requires a number of modifications to the host environment.
The tutorial descibe the required configurations. 
"""

class forte_demo():

    with torch.no_grad():
        extractor = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad")
    pp = pprint.PrettyPrinter(indent=4)

    def configure_pipeline(self, up_through=None):

        p = forte.pipeline.Pipeline()

        p.set_reader(TerminalReader())
        p.add(NLTKSentenceSegmenter())
        p.add(NLTKWordTokenizer())
        p.add(NLTKPOSTagger())
        p.add(NLTKLemmatizer())

        allennlp_config = {
            'processors': "tokenize, pos, srl",
            'tag_formalism': "srl",
            'overwrite_entries': False,
            'allow_parallel_entries': True,
            'srl_url': "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        }
        p.add(AllenNLPProcessor(), config=allennlp_config)

        if up_through == "nlu":
            return p

        boxer_config = {"pack_name": "query"}
        p.add(MultiPackBoxer(), config=boxer_config)
        query_creator_config = {"size": 10}
        p.add(ElasticSearchQueryCreator(), query_creator_config)

        if up_through == "es_query":
            return p

        elastic_search_config = {
            "query_pack_name": "query",
            "index_config":{
                "index_name": "elastic_index",
                "hosts": "localhost:9200",
                "algorithm": "bm25",
            },
            "field": "content",
            "response_pack_name_prefix": "passage",
            "indexed_text_only": False
        }
        p.add(ElasticSearchProcessor(), elastic_search_config)

        if up_through == "retrieval":
            return p

        return p

    def inspect_nlu(self):

        p = self.configure_pipeline(up_through="nlu")
        p.initialize()

        data_pack = next(p.process_dataset())
        for sent in data_pack.get(Sentence):
            print("Tokens created by NLTK:")
            for token in data_pack.get(Token, sent, components=["fortex.nltk.nltk_processors.NLTKWordTokenizer"]):
                print(f" text: {token.text}, pos: {token.pos}, lemma: {token.lemma}")
        print(f"Query: {data_pack.text}")
        print("Semantic role labels created by AllenNLP:")
        for pred in data_pack.get(PredicateLink, sent, components=["fortex.allennlp.allennlp_processors.AllenNLPProcessor"]):
            verb = pred.get_parent()
            noun = pred.get_child()
            print(f" verb: {verb.text}, noun: {data_pack.text[noun.begin:noun.end]}, noun_type: {pred.arg_type}")

    def inspect_es_query(self):
        p = self.configure_pipeline(up_through="es_query")
        p.initialize()
        self.pp.pprint(next(p.process_dataset()).get_pack("query").get_single(Query).value)

    def inspect_retrieval(self):
        p = self.configure_pipeline(up_through="retrieval")
        p.initialize()
        for i, m_pack in enumerate(p.process_dataset()):
            question = m_pack.get_pack('query').text
            print(f"Question: {question}")
            print(f"------Context {i} -----\n")
            for pack in m_pack.packs:
                if pack.pack_name != "query":
                    print(f" {pack.text[:1000]}...")

                print(f"-----Answer----\n")
                result = self.extractor(question=question, context=pack.text)
                print(result['answer'])

if __name__ == "__main__":

    fd = forte_demo()
    exit = False
    while not exit:
        print("Please select from the following options:") 
        print("[1] inspect nlu ") 
        print("[2] inspect es query") 
        print("[3] inspect end-to-end") 
        s = input(">>") 

        if s == "1":
            fd.inspect_nlu()
        elif s == "2":
            fd.inspect_es_query()
        elif s == "3":
            fd.inspect_retrieval()
        elif s == "exit":
            exit = True
        else:
            print("Invalid selection")
    print("goodbye")
