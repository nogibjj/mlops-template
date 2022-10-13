install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
install-tensorflow:
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
	/home/codespace/venv/bin/pip install -r tf-requirements.txt

test:
	python -m pytest -vv --cov=main --cov=mylib test_*.py

format:	
	black *.py 

lint:
	pylint --disable=R,C --ignore-patterns=test_.*?py *.py mylib/*.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	#deploy goes here
		
all: install lint test format deploy
