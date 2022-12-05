#!/usr/bin/env python
"""Create Zero-shot classification command-line tool using Hugging Face's transformers library."""

from transformers import pipeline
import click

# Create a function that reads a file
def read_file(filename):
    with open(filename, encoding="utf-8") as myfile:
        return myfile.read()


# create a function that grabs candidate labels from a file
def read_labels(kw_file):
    return read_file(kw_file).splitlines()


# create a function that reads a file performs zero-shot classification
def classify(text, labels, model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
    classifier = pipeline("zero-shot-classification", model=model)
    results = classifier(text, labels, multi_label=False)
    return results


# create click group
@click.group()
def cli():
    """A cli for zero-shot classification"""


# create a click command that performs zero-shot classification
@cli.command("classify")
@click.argument("filename", default="four-score.m4a.txt")
@click.argument("kw_file", default="keywords.txt")
def classifycli(filename, kw_file):
    """Classify text using zero-shot classification"""
    text = read_file(filename)
    labels = read_labels(kw_file)  # needs to be a sequence
    results = classify(text, labels)
    # print out each label and its score in a tabular format with colors
    for label, score in zip(results["labels"], results["scores"]):
        click.secho(f"{label}\t{score:.2f}", fg="green")


if __name__ == "__main__":
    cli()
