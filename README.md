# Fine-Grained Entity Typing in Hyperbolic Space
Code for the paper ["Fine-Grained Entity Typing in Hyperbolic Space"](https://www.aclweb.org/anthology/W19-4319) published at RepL4NLP @ ACL 2019

Model overview:
<p align="center"><img width="85%" src="img/model.png" /></p>

## Citation
The source code and data in this repository aims at facilitating the study of fine-grained entity typing. If you use the code/data, please cite it as follows:
```
@inproceedings{lopez-etal-2019-fine,
    title = "Fine-Grained Entity Typing in Hyperbolic Space",
    author = "L{\'o}pez, Federico  and
      Heinzerling, Benjamin  and
      Strube, Michael",
    booktitle = "Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-4319",
    pages = "169--180",
}
```

## Dependencies
* ``PyTorch 1.1``
* ``tqdm``
* ``tensorboardX``
* ``pyflann``

A conda environment can be created as well from the ``environment.yml`` file.

To embed the graphs into the different metric spaces the library [Hype](https://github.com/facebookresearch/poincare-embeddings/) was used. 

## Running the code

### 1. Download data
Download and uncompress Ultra-Fine dataset and GloVe word embeddings:
```
./scripts/figet.sh get_data
```

### 2. Preprocess data 
The parameter ``freq-sym`` can be replaced to store different preprocessing configurations: 
```
./scripts/figet.sh preprocess freq-sym
```

### 3. Train model
The name of the preprocessing used in the previous step must be given as a parameter.
```
./scripts/figet.sh train freq-sym
```

### 3. Do inference
```
./scripts/figet.sh inference freq-sym
```


## Acknowledgements
We thank to [Choi et al](https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf) for the release of the Ultra-Fine dataset and [their model](https://github.com/uwnlp/open_type).

## License

[MIT](LICENSE)
