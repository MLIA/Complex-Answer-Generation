import json
import subprocess
import os
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher

class BM25:

    def __init__(self, folder_index, xp_name="default"):
        self.xp_name = xp_name
        self.folder_index = folder_index
    
    @staticmethod 
    def create_index(documents_dict, folder_path):
        documents =\
            [{"id": str(doc_id), "contents": doc_content} for doc_id, doc_content in documents_dict.items()]
        os.makedirs(os.path.join(folder_path,'documents'), exist_ok=True)
        with open(os.path.join(folder_path,'documents/documents.json'), 'w') as tmp_file_index:
            json.dump(documents, tmp_file_index) 

        command_line = 'python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 1 -input '+ os.path.join(folder_path, 'documents/')+' -index '+os.path.join(folder_path, 'pyserini_index/index')+' -storePositions -storeDocvectors -storeRaw'
        os.system(command_line)  


    def rank(self, dataset):
        prediction_set = {}
        index_reader = IndexReader(os.path.join(self.folder_index, 'pyserini_index/index'))
        for query_index, document_index, query in dataset:

            score = index_reader.compute_query_document_score(document_index, query)
            if query_index not in prediction_set:
                prediction_set[query_index] = {}
            prediction_set[query_index][document_index] = score    

        return prediction_set
    
    def rank_corpus(self, queries, top_k=100):
        searcher = SimpleSearcher(os.path.join(self.folder_index, 'pyserini_index/index'))
        searcher.set_bm25()
        prediction_set = {}
        for query_id, query_text in queries.items():
            scores = searcher.search(query_text, k=top_k)
            prediction_set[query_id] ={score.docid: score.score for score in scores}
        return prediction_set


    @staticmethod
    def test():
        dataset = [
            (1, 'A84E6F', 'How to cook beef ?'),
            (2, 'C89QS5', 'tutorial on pytorch lightning')
        ]
        documents = {
            'A84E6F': 'tutorial: Put 1 onion, cut into 8 wedges, and 500g carrots, halved lengthways, into a roasting tin and sit the beef on top, then cook for 20 mins.',
            'C89QS5': 'Learn with Lightning · PyTorch Lightning Training Intro · Automatic Batch Size Finder · Automatic Learning Rate Finder · Exploding And Vanishing Gradients.',
            '22T4AF': 'Be aware whether a method takes or returns analyzed or unanalyzed terms.'
        }

        queries = {
            "1":'How to cook beef ?',
            "2": 'tutorial on pytorch lightning'
        }
        BM25.create_index(documents, './pyserini_bm25_testing')
        ranker = BM25('./pyserini_bm25_testing')
        print(ranker.rank(dataset))
        print(ranker.rank_corpus(queries))
