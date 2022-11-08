"""This is the data loading and processing function for the trainer"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def download_data(data_pathway):
    """This function will take the data_pathway and return a dataframe with the data."""
    dataset = load_dataset(data_pathway)
    df = pd.DataFrame(dataset["train"])
    return df


def dataset_sampler(dataframe):
    """This function will take the dataframe and return a sample and columns."""
    return dataframe.sample(5), dataframe.columns


def available_characters(char_col, quotes_df):
    """Returns a list of the available characters in the dataset from the specified column.
    Args:
        char_col (str): The name of the column containing the character names.
    Returns:
        list: Value counts of the available characters knee-capped at 40 lines.
    """

    # get the unique characters
    characters = quotes_df[char_col].value_counts()

    # return the list of characters for those with more than 40 quotes
    return characters[characters > 40]


def context_builder(character, character_col, line_col, quotes_df):
    """This function will take the character, line column, and dataframe and return a list of the character's lines with context.

    Args:
        character (str): The name of the character to train the model on.
        character_col (str): The name of the column containing the character names.
        line_col (str): The name of the column containing the character lines.
        quotes_df (pandas.DataFrame): The dataframe containing the quotes.

    Returns:
        quotes_context_df (pandas.DataFrame): The dataframe containing the quotes with seven preceeding context quotes."""

    # make an empty dataframe to hold the quotes and context
    context_df = pd.DataFrame(
        columns=[
            "quote",
            "context/0",
            "context/1",
            "context/2",
            "context/3",
            "context/4",
            "context/5",
        ]
    )

    # iterate through the quotes dataframe and add the quotes and context to the quotes_context_df starting with row 7
    for i in range(7, len(quotes_df)):
        # if the character in the row matches the character we're looking for
        if quotes_df[character_col][i] == character:
            # concat the result to the context_df
            context_df = pd.concat(
                [
                    context_df,
                    pd.DataFrame(
                        {
                            "quote": [quotes_df[line_col][i]],
                            "context/0": [quotes_df[line_col][i - 1]],
                            "context/1": [quotes_df[line_col][i - 2]],
                            "context/2": [quotes_df[line_col][i - 3]],
                            "context/3": [quotes_df[line_col][i - 4]],
                            "context/4": [quotes_df[line_col][i - 5]],
                            "context/5": [quotes_df[line_col][i - 6]],
                        }
                    ),
                ]
            )

    # return the quotes_context_df
    return context_df


def data_setup():
    """Function to collect user input at desired intervals and return outputs"""

    # Ask user to define the location of the data on Hugging Face
    data_pathway = input(
        "\n>>> Please enter the location of the data on Hugging Face in this format: username/datasetname: "
    )

    # download the data
    df = download_data(data_pathway)

    # sample the data
    dataset_sample, dataset_cols = dataset_sampler(df)

    # show a sample of the dataset
    print(f"\n>>> Here is a sample of the dataset: \n{dataset_sample}")

    # Ask the user to confirm that the data is correct
    data_correct = input("\n>>> Does the data look correct? (y/n): ")

    # If the data is correct, ask the user to define the columns for setting the training structure
    if data_correct == "y":
        print(f"\n>>> The available columns are: {dataset_cols}")

        # Ask the user to define the column for the speaker and the column for the quote
        speaker_col = input("\n>>> Please enter the column for the speaker: ")
        quote_col = input(">>> Please enter the column for the quote: ")

        # show the user the available character speakers
        speakers = available_characters(speaker_col, df)
        print(f"\n>>> The available characters are: \n{speakers}")

        # Ask the user to define the character to train the model on
        character = input("\n>>> Please enter the character to train the model on: ")

        # send the character, character column, quote column, and df to the context builder
        context_df = context_builder(character, speaker_col, quote_col, df)

        # print a quick sample and confirm it looks right
        print(f"\n>>> Here is a sample of the context: \n{context_df}")
        context_correct = input("\n>>> Does the context look correct? (y/n): ")

        # if the context is correct, make the train test split
        if context_correct == "y":

            trn_df, val_df = train_test_split(context_df, test_size=0.2)

            return trn_df, val_df

        # if the context is not correct, return to the beginning of the function
        else:
            data_setup()

    # If the data is not correct, exit the program
    else:
        data_setup()
