import os
from typing import List
import json
import argparse
import logging
import pandas as pd
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
    queries = [
        "Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?",
        "Which individual is implicated in both inflating the value of a Manhattan apartment to a figure not yet achieved in New York City's real estate history, according to 'Fortune', and is also accused of adjusting this apartment's valuation to compensate for a loss in another asset's worth, as reported by 'The Age'?",
        "Who is the figure associated with generative AI technology whose departure from OpenAI was considered shocking according to Fortune, and is also the subject of a prevailing theory suggesting a lack of full truthfulness with the board as reported by TechCrunch?"
    ]

    # For Evaluation
    answers = [
        "Sam Bankman-Fried",
        "Donald Trump",
        "Sam Altman"
    ]

    gold_docs = [
        [   
            df.loc[df['title'] == "The FTX trial is bigger than Sam Bankman-Fried", 'content'].iloc[0],
            df.loc[df['title'] == "SBF\u2019s trial starts soon, but how did he \u2014 and FTX \u2014 get here?", 'content'].iloc[0]
        ],
        [
            df.loc[df['title'] == "Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president", 'content'].iloc[0],
            df.loc[df['title'] == "The $777 million surprise: Donald Trump is getting richer", 'content'].iloc[0]
        ],
        [
            df.loc[df['title'] == "OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did'", 'content'].iloc[0],
            df.loc[df['title'] == "WTF is going on at OpenAI? We have theories", 'content'].iloc[0]
        ]
    ]

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers))

if __name__ == "__main__":
    main()
