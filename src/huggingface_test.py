#%%
import os
from dotenv import load_dotenv

from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# %%
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

result = client.token_classification(
    "Das Thema müssen wir Linguisten mit euch widerspenstigen InformatikerInnen noch klären",
    model="gfkpth/bert-apc-detector-ger",
)


# %%

from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="gfkpth/bert-apc-detector-ger",
    aggregation_strategy='first'
)

text="Das Thema müssen wir ErzieherInnen von euch widerspenstigen Kindern noch mit dem Schulamt klären. Eigentlich haben wir Tiere nur im Zoo gesehen. Das haben wir Tiere aber geschickt gelöst."

output_text=text
results = pipe(text)

for r in sorted(results, key=lambda x: x['start'], reverse=True):
    start, end, label = r['start'], r['end'], r['entity_group']
    output_text = output_text[:start] + f"[{label}: " + output_text[start:end] + "]" + output_text[end:]

print('Without nested tags:')
print(output_text)

# trying nested tags
insertions = []

for span in results:
    start, end, label = span['start'], span['end'], span['entity_group']
    insertions.append((start, f"[{label}: "))
    insertions.append((end, "]"))

# Sort insertions in reverse order of positions
insertions.sort(key=lambda x: x[0], reverse=True)

# Apply insertions
output_text = text
for pos, s in insertions:
    output_text = output_text[:pos] + s + output_text[pos:]


print('With nested tags')
print(output_text)
