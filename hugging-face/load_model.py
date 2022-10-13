"""Loading a model from a file."""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("summarizeApp")
tokenizer = AutoTokenizer.from_pretrained("summarizeApp")
#open the file with utf-8 encoding
with open("input.txt", encoding="utf-8") as f:
    text = f.read()

input_ids = tokenizer.encode(text, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)