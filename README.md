# Enhancing Machine Translation with Dependency-Aware Self-Attention

This is the implementation of the approaches described in the paper:
> Emanuele Bugliarello and Naoaki Okazaki. [Enhancing Machine Translation with Dependency-Aware Self-Attention](https://arxiv.org/abs/1909.03149). In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, July 2020.

We provide the code for reproducing our results, as well as translation outputs of each model.

## Requirements
The requirements can be installed by setting up a conda environment: <br>
`conda env create -f environment.yml` followed by `source activate pascal`

## Data Preparation
The pre-processing steps for each model in each data set can be found in the corresponding `experiments/` folder, and rely on our code (`scripts/`) as well as on third-party software (`tools/`).

## Training
Scripts for training each model are provided in the corresponding data set folder in `experiments/` (e.g., [`experiments/wmt16en2de/transformer/train.sh`](experiments/wmt16en2de/transformer/train.sh)).

Note that we trained our models on a SGE cluster. 
To run our training experiments, submit (`qsub`) the corresponding `train.sge` file for a given experiment. 
It calls the `train.sh` file associated in its directory.

## Evaluation

Similarly, you can use the corresponding `eval.sh` and `eval.sge` files to evaluate a model.

## Description of this repository
- `experiments/`<br>
  Contains code to reproduce our results. For each data set, the following files are used to prepare the data:
  - `prepare_data.sh`: Google's pre-processing steps for the Transformer model
  - `prepare_filt_data.sh`: use [langdetect](https://pypi.org/project/langdetect/) to remove sentences in languages that do not match source or target ones
  - `prepare_lin_parses.sh`: extract linearized parses for the multi-task approach of [Currey and Heatfield (WMT'19)](https://www.aclweb.org/anthology/W19-5203/).
  - `prepare_tags_label.sh`: extract dependency labels following the approach of [Sennrich and Haddow (WMT'16)](https://www.aclweb.org/anthology/W16-2209/)
  - `prepare_tags_mean.sh`: extract dependency heads and map them to mean/middle position of the parent's sub-word units
  - `prepare_tags_root.sh`: extract dependency heads and map them to first position (root) of the parent's sub-word units
  - `binarize_*.sh` files: convert text data into binary files used by Fairseq

- `fairseq/`<br>
  Our code is based on a fork of [Fairseq](https://github.com/pytorch/fairseq) (commit ID can be found in `VERSION.md`).
  Here, we introduce a new `tags-translation` task to accept two source files (words and syntactic tags). 
  This is implemented through the following files:
  - `data/tags_language_pair_dataset.py`
  - `models/fairseq_tags_encoder.py`
  - `models/fairseq_model.py`
  - `tasks/tags_translation.py`
  
  We also implement the following dependency-aware Transformer models:
  - Pascal: Parent-Scaled Self-Attention mechanism
    - `models/pascal_transformer.py`
    - `modules/multihead_pascal.py`
  - LISA: adaptation of [Strubell et al. (EMNLP'18)](https://www.aclweb.org/anthology/D18-1548/) to sub-word units
    - `models/lisa_transformer.py`
    - `modules/multihead_lisa.py`
    - `criterions/lisa_cross_entropy.py`
  - TagEmb (S&H): adaptation of [Sennrich and Haddow (WMT'16)](https://www.aclweb.org/anthology/W16-2209/) for the Transformer model
    - `models/tagemb_transformer.py`
  - Multi-Task (C&H): our implementation of [Currey and Heatfield (WMT'19)](https://www.aclweb.org/anthology/W19-5203/)
    - This is a data augmentation technique, so no new models were created

- `scripts/`: data preparation scripts for extracting syntactic tags
- `tools/`: third-party and own software used in pre-processing (e.g., Moses and BPE) as well as evaluation (e.g., RIBES)

## License
This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/models or ideas useful in your research, please consider citing the paper:
```
@inproceedings{bugliarello-okazaki-2020-enhancing,
  title={Enhancing Machine Translation with Dependency-Aware Self-Attention},
  author={Bugliarello, Emanuele and Okazaki, Naoaki},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  month={jul},
  year={2020},
  publisher={Association for Computational Linguistics},
  url={https://arxiv.org/abs/1909.03149}
}
```
