#!/usr/bin/env python3

"""
Main cli or app entry point
"""
import yake
import click

# create a function that reads a file
def read_file(filename):
    with open(filename, encoding="utf-8") as myfile:
        return myfile.read()


# def extract keywords
def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return keywords


# create a function that makes hash tags
def make_hashtags(keywords):
    hashtags = []
    for keyword in keywords:
        hashtags.append("#" + keyword[0].replace(" ", ""))
    return hashtags

#create a function returns only the top N keywords
def top_n_keywords(keywords, n=5):
    return keywords[:n]

@click.group()
def cli():
    """A cli for keyword extraction"""

#print the top N keywords
@cli.command("extract")
@click.argument("filename", default="text.txt")
@click.option("--n", default=5, help="Number of keywords to extract")
def extract(filename, n):
    """Extract keywords from a file"""
    text = read_file(filename)
    keywords = extract_keywords(text)
    top_keywords = top_n_keywords(keywords, n)
    for keyword in top_keywords:
        print(keyword[0])


@cli.command("hashtags")
@click.argument("filename", default="text.txt")
def hashtagscli(filename):
    """Extract keywords from a file and make hashtags"""
    text = read_file(filename)
    keywords = extract_keywords(text)
    hashtags = make_hashtags(keywords)
    click.echo(hashtags)


if __name__ == "__main__":
    cli()