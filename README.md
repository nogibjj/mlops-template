[![HF Fine-Tune CI](https://github.com/nogibjj/hugging-face-gpt-trainer/actions/workflows/cicd.yml/badge.svg)](https://github.com/nogibjj/hugging-face-gpt-trainer/actions/workflows/cicd.yml)

# Welcome

You've found your way to a pretty funny little project. Below is a visual description of what happens here, and there is [also a video.](https://youtu.be/x7Pnf6SNRYA)

<img src="https://github.com/andrewkroening/hugging-face-gpt-trainer/blob/1b948c4380fb112d75f2f61e7d19266a6652ef47/utils/Reduced_Workflow.png" alt="Workflow" width="700"/>

Here's the gist:

* We take a specified gnerative text model, such as GPT-2

* We define a dataset and do some transformations

* We fine-tune the model on the dataset

* We push the model to the Hugging Face model repository

* And 'pylint' does absolutely NOT like the way our code looks

## Contents

There are a few quick things to note about this repository:

* The main files you'll need are in the main directory and begin 'fine_tune_'. The place to start is 'fine_tune_main.py' and once you're done fine-tuning you can use 'fine_tune_test_chat.py' to chat with your model.

* The 'cached', 'model', and 'runs' directories contain information used to fine-tune the model. These directories are included in the '.gitignore' file.

* 'utils' has a few helpful tools to poke around the GPU environment this codespace relies upon.

## Acknowledgements

There are several sources of inspiration and insight for the project that spawned this model. I'd like to recognize them up front:

* The [Microsoft DialoGPT-Medium](https://huggingface.co/microsoft/DialoGPT-medium?text=Hi.) model page was very insightful for getting stated.

* Lynn Zheng [r3dhummingbird](https://huggingface.co/r3dhummingbird/DialoGPT-medium-joshua?text=Hey+my+name+is+Thomas%21+How+are+you%3F) put together one heck of an awesome tutorial on how to fine-tune GPT-2 for conversational purposes. I used her tutorial as a starting point for this project. Check out the [Github repo here.](https://github.com/RuolinZheng08/twewy-discord-chatbot)

* [This article](https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30) was also very insightful, written by Rostyslav Neskorozhenyi.

* From a lineage standpoint, it looks like Nathan Cooper kicked this whole thing off with this [notebook.](https://github.com/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb)

* Noah Gift figured out a few of the big pieces in [this repository.](https://github.com/nogibjj/hugging-face-tutorial-practice)

* I'd be remiss if I also didn't mention Hugging Face's own support [documentation](https://huggingface.co/transformers/v2.0.0/examples.html#gpt-2-gpt-and-causal-language-modeling) and team. All around great.

## GPT-2 Model Information

This model is based on a GPT-2 model which was fine-tuned on a Hugging Face dataset. It is intended largely as an illustrative example and is not intended to be used for any serious purpose. It's trained on a movie script for goodness' sake.

Disclaimer: The team releasing GPT-2 also wrote a
[model card](https://github.com/openai/gpt-2/blob/master/model_card.md) for their model. Content from this model card
has been written by the Hugging Face team to complete the information they provided and give specific examples of bias.

## Model description

This model uses GPT-2 Medium as a base model and was fine-tuned using scripts from the original (and best) Star Wars Trilogy. In this particular case, it was fine-tuned on Luke Skywalker's 900-some lines. This is not a lot, and thus the model should not be assumed to have serious integrity. It's just a fun little project.

## Intended uses & limitations

This model is intended to be used for fun and entertainment. Don't take it too seriously.

### Ways to use

You can always chat with the model directly on the Hugging Face website. Just click the "Chat" button on the right side of the model page.

If you want to use the model in your own project, I recommend you train it better using much more data.

## Fine-tuning data

The script to generate this model takes a Hugging Face data set in this approximate format:

| Speaker | Text |
| --- | --- |
| Luke | Hello there. |
| Han | General Kenobi. |
| Luke | You are a bold one. |

The script then asks the user to define parameters for making the dataset and proceeding to fine-tuning. The actual dataset for this model can be found [here.](andrewkroening/Star-wars-scripts-dialogue-IV-VI)
