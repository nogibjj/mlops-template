## Bento Exploration

`bentoml models list`

### Very clean format for dealing with models

`ls /home/codespace/bentoml/models/iris_clf/ecde3adktowieaan/`
`model.yaml` and `saved_model.pkl`

### Export the model

`bentoml models export iris_clf:latest .`

### Try out Yatai

* Must install minikube

https://github.com/bentoml/Yatai

`minikube start --cpus 4 --memory 4096`

Check if it is running:

`minikube status`

Double check context:

`kubectl config current-context`

Enable ingress

`minikube addons enable ingress`

Then install:

`bash <(curl -s "https://raw.githubusercontent.com/bentoml/yatai/main/scripts/quick-install-yatai.sh")`