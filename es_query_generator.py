from typing import Any, Dict, Tuple
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base import QueryProcessor
from fortex.elastic.elastic_search_processor import ElasticSearchProcessor
from composable_source.utils.utils import query_preprocess


class ElasticSearchQueryCreator(QueryProcessor):
    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "size": 1000,
            "field": "content",
            "query_pack_name": "query"
        }) 
        return config

    def _process_query(self, input_pack: MultiPack) -> Tuple[DataPack, Dict[str, Any]]:
        query_pack = input_pack.get_pack(self.configs.query_pack_name)
        query_pack.pack_name = self.configs.query_pack_name
        query = self._build_query_nlp(query_pack)
        return query_pack, query

    def _build_query_nlp(self, input_pack: DataPack) -> Dict[str, Any]:
        query, arg0, arg1, verb, _, is_answer_arg0 = query_preprocess(input_pack)
        if not arg0 or not arg1:
            processed_query = query
        if is_answer_arg0 is None:
            processed_query = f'{arg0} {verb} {arg1}'.lower()
        elif is_answer_arg0:
            processed_query = f'{arg1} {verb}'.lower()
        else:
            processed_query = f'{arg0} {verb}'.lower()
        return {
            "query": {
                "match_phrase": {
                    self.configs.field: {
                        "query": processed_query,
                        "slop": 10
                    }
                }
            },
            "size": self.configs.size
        }