[![CI](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml)
[![Codespaces Prebuilds](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds)

## Template for MLOPs projects with GPU

1. First thing to do on launch is to open a new shell and verify virtualenv is sourced.

Things included are:

* `Makefile`

* `Pytest`

* `pandas`

* `Pylint`

* `Dockerfile`

* `GitHub copilot`

* `jupyter` and `ipython` 

* Most common Python libraries for ML/DL and Hugging Face

* `githubactions` 

### Verify GPU works

The following examples test out the GPU

* run pytorch training test: `python utils/quickstart_pytorch.py`
* run pytorch CUDA test: `python utils/verify_cuda_pytorch.py`
* run tensorflow training test: `python utils/quickstart_tf2.py`
* run nvidia monitoring test: `nvidia-smi -l 1` it should show a GPU
* run whisper transcribe test `./utils/transcribe-whisper.sh` and verify GPU is working with `nvidia-smi -l 1`

Additionally, this workspace is setup to fine-tune Hugging Face

![fine-tune](https://user-images.githubusercontent.com/58792/195709866-121f994e-3531-493b-99af-c3266c4e28ea.jpg)


`python hf_fine_tune_hello_world.py` 

### Used in Following Projects

Used as the base and customized in the following Duke MLOps and Applied Data Engineering Coursera Labs:

* [MLOPs-C2-Lab1-CICD](https://github.com/nogibjj/Coursera-MLOPs-Foundations-Lab-1-CICD)
* [MLOps-C2-Lab2-PokerSimulator](https://github.com/nogibjj/Coursera-MLOPs-Foundations-Lab-2-poker-simulator)
* [MLOps-C2-Final-HuggingFace](https://github.com/nogibjj/Coursera-MLOps-C2-Final-HuggingFace)
* [Coursera-MLOps-C2-lab3-probability-simulations](Coursera-MLOps-C2-lab3-probability-simulations)
* [Coursera-MLOps-C2-lab4-greedy-optimization](https://github.com/nogibjj/Coursera-MLOps-C2-lab4-greedy-optimization)
### References

* [Building Cloud Computing Solutions at Scale Specialization](https://www.coursera.org/specializations/building-cloud-computing-solutions-at-scale)
* [Python, Bash and SQL Essentials for Data Engineering Specialization](https://www.coursera.org/learn/web-app-command-line-tools-for-data-engineering-duke)
* [Implementing MLOps in the Enterprise](https://learning.oreilly.com/library/view/implementing-mlops-in/9781098136574/)
* [Practical MLOps: Operationalizing Machine Learning Models](https://www.amazon.com/Practical-MLOps-Operationalizing-Machine-Learning/dp/1098103017)
* [Coursera-Dockerfile](https://gist.github.com/noahgift/82a34d56f0a8f347865baaa685d5e98d)
