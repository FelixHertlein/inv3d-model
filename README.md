🚀 Good news! We have have created a demo showcasing the capabilities of the model GeoTrTemplateLarge whithin the full document refinement pipeline. [Check it out here!](https://felixhertlein.de/docrefine/home)

# Inv3D - Models

This repository contains the models and inference, training and evaluation code of our [paper](https://link.springer.com/article/10.1007/s10032-023-00434-x) which has been accepted at the International Conference on Document Analysis and Recognition ([ICDAR](https://icdar2023.org/)) 2023.

For more details see our project page [project page](https://felixhertlein.github.io/inv3d).

## Usage

### VS-Code Devcontainer

We highly recommend to use the provided Devcontainer to make the usage as easy as possible:

- Install [Docker](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/)
- Install VS Code Devcontainer extension `ms-vscode-remote.remote-containers`
- Clone the repository
  ```shell
  git clone https://github.com/FelixHertlein/inv3d-model.git
  ```
- Press `F1` (or `CTRL + SHIFT + P`) and select `Dev Containers: Rebuild and Reopen Container`
- Go to `Run and Debug (CTRL + SHIFT + D)` and press the run button, alternatively press `F5`

## Inference

### Start the inference

`python3 inference.py --model geotr_template@inv3d --dataset inv3d_real --gpu 0`

### Models

The models will be downloaded automatically before the inference starts.

Available models are:

- geotr@doc3d
- geotr@inv3d
- geotr_template@inv3d
- geotr_template_large@inv3d

### Datasets

#### Inv3DReal

Inv3DReal is part of this repository and can be used by passing `inv3d_real` as the dataset argument.

#### Custom dataset

To unwarp your own data, you can mout your data inside the container using the `.devcontainer/devcontainer.json` config.

Mount your data folder to `/workspaces/inv3d-model/input/YOUR_DATA_DIRECTORY`.
Make sure, all images start with the prefix `image_` and the corresponding templates (only for template-based models) with the prefix `template_`.

### Output: Unwarped images

All unwarped images are placed in the `output` folder.

## Training

### Training datasets

#### Inv3D

Download Inv3D [here](https://publikationen.bibliothek.kit.edu/1000161884), combine all downloads and mount it using the devcontainer.json, such that the file tree looks as follows:

```
input/inv3d
|-- data
|   |-- test
|   |-- train
|   |-- val
|   `-- wc_min_max.json
|-- log.txt
`-- settings.json
```

#### Doc3D

Download Doc3D [here](https://github.com/cvlab-stonybrook/doc3D-dataset), combine all downloads and mount it using the devcontainer.json, such that the file tree looks as follows:

```
input/doc3d
|-- alb
|-- augtexnames.txt
|-- bm
|-- dmap
|-- img
|-- norm
|-- real
|-- recon
|-- test.txt
|-- train.txt
|-- uv
|-- val.txt
`-- wc
```

### Start a new training

`python3 train.py --model geotr_template --dataset inv3d --version v1 --gpu 0 --num_workers 32 `

### Resume a training

`python3 train.py --model geotr_template --dataset inv3d --version v1 --gpu 0 --num_workers 32 --resume`

### Training output

```
models/TRAINED_MODEL/
|-- checkpoints
|   |-- checkpoint-epoch=00-val_mse_loss=0.0015.ckpt
|   `-- last.ckpt
|-- logs
|   |-- events.out.tfevents.1698250741.d6258ba74799.433.0
|   |-- ...
|   `-- hparams.yaml
`-- model.py
```

### Help

```
train.py [-h]
--model MODEL
--dataset DATASET
[--version VERSION]
--num_workers NUM_WORKERS
[--fast_dev_run]
[--model_kwargs MODEL_KWARGS]
[--resume]

Training script

options:
  -h, --help            show this help message and exit
  --model {dewarpnet_bm,dewarpnet_joint,dewarpnet_wc,geotr,geotr_template,geotr_template_large,identity}
                        Select the model for training.
  --dataset {doc3d,doc3d_real,doc3d_test,empty,inv3d,inv3d_real,inv3d_real_tplrandom,inv3d_real_tplstruct,inv3d_real_tpltext,inv3d_real_tplwhite,inv3d_test,inv3d_test_tplrandom,inv3d_test_tplstruct,inv3d_test_tpltext,inv3d_test_tplwhite,inv3d_tplstruct,inv3d_tpltext,inv3d_tplwhite}
                        Select the dataset to train on.
  --version VERSION     Specify a version id for given training. Optional.
  --gpu GPU
                        The index of the GPU to use as an integer.
  --num_workers NUM_WORKERS
                        The number of workers as an integer.
  --fast_dev_run        Enable fast development run (default is False).
  --model_kwargs MODEL_KWARGS
                        Optional model keyword arguments as a JSON string.
  --resume              Resume from a previous run (default is False).
```

## Evaluation

The evaluation contains MS-SSIM, LPIPS, CER and ED. LD is not in this repo since it requires the proprietary software Matlab.

### Start an evaluation

`python3 eval.py --trained_model geotr_template@inv3d@v1 --dataset inv3d_real --gpu 0 --num_workers 16`

Evaluation output:

```
models/TRAINED_MODEL/
|-- eval
|   `-- DATASET
|       |-- examples
|       |   |-- 0
|       |   |   |-- norm_image.png
|       |   |   |-- orig_image.png
|       |   |   |-- out_bm.npz
|       |   |   `-- true_image.png
|       |   |-- ...
|       `-- results.csv
|-- logs
```

### Help

```
eval.py [-h]
--trained_model MODEL
--dataset DATASET
--gpu GPU
--num_workers NUM_WORKERS

Evaluation script

options:
  -h, --help            show this help message and exit
  --trained_model {the model in the models directory}
                        Select the model for evaluation.
  --dataset {doc3d,doc3d_real,doc3d_test,empty,inv3d,inv3d_real,inv3d_real_tplrandom,inv3d_real_tplstruct,inv3d_real_tpltext,inv3d_real_tplwhite,inv3d_test,inv3d_test_tplrandom,inv3d_test_tplstruct,inv3d_test_tpltext,inv3d_test_tplwhite,inv3d_tplstruct,inv3d_tpltext,inv3d_tplwhite}
                        Select the dataset to evaluate on.
  --gpu GPU             The index of the GPU to use for training.
  --num_workers NUM_WORKERS
                        The number of workers as an integer.
```

## Citation

If you use the code of our paper for scientific research, please consider citing

```latex
@article{hertlein2023inv3d,
	title        = {Inv3D: a high-resolution 3D invoice dataset for template-guided single-image document unwarping},
	author       = {Hertlein, Felix and Naumann, Alexander and Philipp, Patrick},
	year         = 2023,
	journal      = {International Journal on Document Analysis and Recognition (IJDAR)},
	publisher    = {Springer},
	pages        = {1--12}
}
```

## Acknowledgements

The model GeoTr is part of [DocTr](https://github.com/fh2019ustc/DocTr). GeoTrTemplate is based on GeoTr.

## Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>

## License

This project is licensed under [MIT](LICENSE).
