import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

def main():
    docs_paths = ["sam1.txt", "sam2.txt", "trump1.txt", "trump2.txt", "yes1.txt", "yes2.txt"]

    # Prepare datasets and evaluation
    docs = []

    for path in docs_paths:
        with open(path, "r") as f:
            content = f.read()
            docs.append(content)

    save_dir = 'outputs/openai'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'openai/gpt-4o'  # Any OpenAI model name
    embedding_model_name = 'nvidia/NV-Embed-v2'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    llm_base_url = 'https://models.github.ai/inference' # LLM API endpoint

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        llm_base_url=llm_base_url)

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?",
        "Which individual is implicated in both inflating the value of a Manhattan apartment to a figure not yet achieved in New York City's real estate history, according to 'Fortune', and is also accused of adjusting this apartment's valuation to compensate for a loss in another asset's worth, as reported by 'The Age'?",
        "Do the TechCrunch article on software companies and the Hacker News article on The Epoch Times both report an increase in revenue related to payment and subscription models, respectively?"
    ]

    # For Evaluation
    answers = [
        ["Sam Bankman-Fried"],
        ["Donald Trump"],
        ["Yes"]
    ]

    gold_docs = [
        docs[0:2],
        docs[2:4],
        docs[4:]
    ]

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers))

if __name__ == "__main__":
    main()
