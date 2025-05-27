docs_paths = ["sam1.txt", "sam2.txt", "trump1.txt", "trump2.txt", "yes1.txt", "yes2.txt"]

# Prepare datasets and evaluation
docs = []

for path in docs_paths:
    with open(path, "r") as f:
        content = f.read()
        docs.append(content)

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

print(gold_docs)