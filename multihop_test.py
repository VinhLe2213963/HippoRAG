import os
from typing import List
import json
import argparse
import logging
import pandas as pd
import json
from src.hipporag import HippoRAG

def main():
    df = pd.read_csv("multihoprag_corpus_summary.csv")

    # Prepare datasets and evaluation
    docs = list(df['content'])

    save_dir = 'outputs/openai'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'gemini-2.0-flash'  # Any OpenAI model name
    embedding_model_name = 'nvidia/NV-Embed-v2'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    llm_base_url = 'https://generativelanguage.googleapis.com/v1beta/' # LLM API endpoint
    # https://models.github.ai/inference      openai/gpt-4o-mini
    # https://api.deepseek.com        deepseek-chat
    # https://generativelanguage.googleapis.com/v1beta/
    

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        llm_base_url=llm_base_url
                        )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    with open("MultiHopRAG.json", "r") as f:
        content = json.load(f)

    start = 60
    end = 80

    content = content[start:end]

    queries = [data['query'] for data in content]

    # For Evaluation
    answers = [data['answer'] for data in content]

    gold_docs = []

    for data in content:
        docs = []
        for evidence in data['evidence_list']:
            match = df.loc[df['title'] == evidence['title'], 'content']
            if not match.empty:
                docs.append(match.iloc[0])
            else:
                docs.append(None)  # or skip / handle as needed
        gold_docs.append(docs)

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers))

if __name__ == "__main__":
    main()
