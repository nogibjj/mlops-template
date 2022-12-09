#!/usr/bin/env python
# """Create OpenAI Whisper command-line tool using Hugging Face's transformers library."""

from transformers import pipeline
import click


# Create a function that reads a sample audio file and transcribes it using openai's whisper
def traudio(filename, model="openai/whisper-tiny.en"):
    with open(filename, "rb") as f:
        _ = f.read()  # this needs to be fixed
    print(f"Transcribing {filename}...")
    pipe = pipeline("automatic-speech-recognition", model=model)
    results = pipe(filename)
    return results


# create click group
@click.group()
def cli():
    """A cli for openai whisper"""


# create a click command that transcribes
@cli.command("transcribe")
@click.option(
    "--model", default="openai/whisper-tiny.en", help="Model to use for transcription"
)
@click.argument("filename", default="utils/four-score.m4a")
def whispercli(filename, model):
    """Transcribe audio using openai whisper"""
    results = traudio(filename, model)
    # print out each label and its score in a tabular format with colors
    for result in results:
        click.secho(f"{result['text']}", fg="green")


if __name__ == "__main__":
    cli()
