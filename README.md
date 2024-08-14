## Description of documents
The file mainly contains three parts: models,unsupervised_TU, and requirements.txt.
* models:Different encoder models.
* unsupervised_TU: Contrastive learning models and property inference attacks.
* requirements.txt: The dependencies.

## Dependencies

The dependencies are shown in requirement.txt, and you can install the environment using the following commands

```
pip install -r requirements.txt
```

## Training Gclmodel

```
./go_.sh $GPU_ID $DATASET_NAME 
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/),you can use MUTAG, PROTEINS, AIDS, DD, NCI1, ```$GPU_ID``` is the lanched GPU ID



## Implements attribute inference attacks

```
./go_attack.sh $GPU_ID $DATASET_NAME 
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/),you can use MUTAG, PROTEINS, AIDS, DD, NCI1, ```$GPU_ID``` is the lanched GPU ID




