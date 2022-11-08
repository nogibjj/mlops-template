---
language: en
tags:
- conversational

license: cc
---


# GPT-2

This model is based on a GPT-2 model which was fine-tuned on a Hugging Face dataset. It is intended largely as an illustrative example and is not intended to be used for any serious purpose. It's trained on a movie script for goodness' sake.

Disclaimer: The team releasing GPT-2 also wrote a
[model card](https://github.com/openai/gpt-2/blob/master/model_card.md) for their model. Content from this model card
has been written by the Hugging Face team to complete the information they provided and give specific examples of bias.

## Acknowledgements

There are several sources of inspiration and insight for the project that spawned this model. I'd like to recognize them up front:

* The [Microsoft DialoGPT-Medium](https://huggingface.co/microsoft/DialoGPT-medium?text=Hi.) model page was very insightful for getting stated.

* Lynn Zheng [r3dhummingbird](https://huggingface.co/r3dhummingbird/DialoGPT-medium-joshua?text=Hey+my+name+is+Thomas%21+How+are+you%3F) put together one heck of an awesome tutorial on how to fine-tune GPT-2 for conversational purposes. I used her tutorial as a starting point for this project. Check out the [Github repo here.](https://github.com/RuolinZheng08/twewy-discord-chatbot)

* [This article](https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30) was also very insightful. Written by Rostyslav Neskorozhenyi.

* From a lineage standpoint, it looks like Nathan Cooper kicked this whole thing off with this [notebook.](https://github.com/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb)

* Noah Gift figured out a few of the big pieces in [this repository.](https://github.com/nogibjj/hugging-face-tutorial-practice)

* I'd be remiss if I also didn't mention Hugging Face's own support [documentation](https://huggingface.co/transformers/v2.0.0/examples.html#gpt-2-gpt-and-causal-language-modeling) and team. All around great.

## Model description

This model uses GPT-2 Medium as a base model and was fine-tuned using scripts from the original (and best) Star Wars Trilogy. In this particular case, it was fine-tuned on Luke Skywalker's 900-some lines. This is not a lot, and thus the model should not be assumed to have serious integrity. It's just a fun little project.

## Intended uses & limitations

This model is intended to be used for fun and entertainment. Don't take it too seriously.

### Ways to use

You can always chat with the model directly on the Hugging Face website. Just click the "Chat" button on the right side of the model page.

If you want to use the model in your own project, I recommend you train it better using much more data.

To access the GitHub repository I used to train this model, click [here](https://github.com/nogibjj/hugging-face-gpt-trainer/tree/gpt-fine-tune)

## Fine-tuning data

The script to generate this model takes a Hugging Face data set in this approximate format:

| Speaker | Text |
| --- | --- |
| Luke | Hello there. |
| Han | General Kenobi. |
| Luke | You are a bold one. |

The script then asks the user to define parameters for making the dataset and proceeding to fine-tuning. The actual dataset for this model can be found [here.](andrewkroening/Star-wars-scripts-dialogue-IV-VI)
