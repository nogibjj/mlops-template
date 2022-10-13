"""
After running zip it to save
zip -r summarizeApp.zip summarizeApp 
"""

from transformers import pipeline

model = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    revision="a4f8f3e",
)
model.save_pretrained("summarizeApp")