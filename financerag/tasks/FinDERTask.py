from typing import Optional
import hashlib
from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata

from datasets import load_dataset 

class FinDER(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="FinDER",
            description="Prepared for competition from Linq",
            reference=None,
            dataset={
                "path": "Linq-AI-Research/FinDER",
                "subset": None,
            },
            type="RAG",
            category="s2p",
            modalities=["text"],
            date=None,
            domains=["Report"],
            task_subtypes=[
                "Financial retrieval",
                "Question answering",
            ],
            license=None,
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="human-generated",
            bibtex_citation=None,
        )
        super().__init__(self.metadata)

    def load_data(self):
        print(f"Loading data manually from {self.metadata.dataset['path']}...")
        try:
            ds = load_dataset(self.metadata.dataset["path"], split="train")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e

        self.corpus = {}
        self.queries = {}
        
        print("Processing documents and queries based on actual schema...")
        
        for row in ds:
            q_id = str(row.get("_id"))
            query_text = row.get("text")
            
            if q_id and query_text:
                self.queries[q_id] = query_text

            refs = row.get("references")
            if isinstance(refs, list):
                for doc_content in refs:
                    if not doc_content:
                        continue
                        
                    doc_hash = hashlib.md5(doc_content.encode('utf-8')).hexdigest()
                    
                    title = doc_content.split('\n')[0] if doc_content else "Financial Report"
                    
                    if doc_hash not in self.corpus:
                        self.corpus[doc_hash] = {
                            "title": title,
                            "text": doc_content
                        }

        print(f"Loaded {len(self.corpus)} unique documents into Corpus.")
        print(f"Loaded {len(self.queries)} queries.")
        
        if len(self.queries) == 0:
            print("Warning: Still found 0 queries. Please check the dataset schema again.")
        else:
            first_q_id = next(iter(self.queries))
            print(f"Example Query [{first_q_id}]: {self.queries[first_q_id]}")