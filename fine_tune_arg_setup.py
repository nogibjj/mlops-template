"""This function asks the user to specify the model they want to use, its tokenizer, and the output direectory/repo they want to use when fine tuning."""


def arg_define():
    """Defines key parameters for the fine tuner before it begins to run.

    Args:
        none

    Returns:
        model_name (str): The name of the model to be fine tuned.
        tokenizer_name (str): The name of the tokenizer to be used with the model.
        output_dir (str): The name of the directory to store the fine tuned model.
        repo_name (str): The name of the repo to store the fine tuned model.
    """

    # show a welcome message
    print(
        "\n>>> Welcome to this Hugging Face GPT-Medium model trainer. \nIt will give you step-by-step prompts to prepare your data for fine-tuning. \nA couple things to note: spelling counts, and you need to use Hugging Face to accomplish this."
    )

    # define the model name
    model_name = input("\n\n>>> What model would you like to fine tune? ")

    # define the tokenizer name
    tokenizer_name = input("\n>>> What tokenizer would you like to use? ")

    # define the output directory
    output_dir = input(
        "\n>>> What directory would you like to store the fine tuned model? "
    )

    # define the repo name
    repo_name = input("\n>>> What repo would you like to store the fine tuned model? ")

    return model_name, tokenizer_name, output_dir, repo_name
