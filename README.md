# Inv3D - Models

This repository contains the models and inference code of our [paper](https://link.springer.com/article/10.1007/s10032-023-00434-x) which has been accepted at the International Conference on Document Analysis and Recognition ([ICDAR](https://icdar2023.org/)) 2023.

For more details see our project page [project page](https://felixhertlein.github.io/inv3d).

## Usage

### VS-Code Devcontainer (recommended)

We highly recommend to use the provided Devcontainer to make the usage as easy as possible:

- Install [Docker](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/)
- Install VS Code Devcontainer extension `ms-vscode-remote.remote-containers`
- Clone the repository
  ```shell
  git clone https://github.com/FelixHertlein/inv3d-model.git
  ```
- Press `F1` (or `CTRL + SHIFT + P`) and select `Dev Containers: Rebuild and Reopen Container`
- Go to `Run and Debug (CTRL + SHIFT + D)` and press the run button, alternatively press `F5`

Inside the container, run

```shell
CUDA_VISIBLE_DEVICES=1 python3 /workspaces/inv3d-model/inference.py --model geotr_template@inv3d --dataset inv3d_real
```

### Docker

TBD

## Models

The models will be downloaded automatically before the inference starts.

Available models are:

- geotr@doc3d
- geotr@inv3d
- geotr_template@inv3d
- geotr_template_large@inv3d

## Datasets

### Inv3DReal

Inv3DReal is part of this repository and can be used by passing `inv3d_real` as the dataset argument.

### Custom dataset

To unwarp your own data, you can mout your data inside the container using the `.devcontainer/devcontainer.json` config.

Mount your data folder to `/workspaces/inv3d-model/input`. 
Make sure, all images start with the prefix `image_` and the corresponding templates (only for template-based models) with the prefix `template_`.

## Unwarped images

All unwarped images are stored in the `output` folder.


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

The model GeoTr is part of [DocTr](https://github.com/fh2019ustc/DocTr).  GeoTrTemplate is based on GeoTr.

## Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>

## License

This project is licensed under [MIT](LICENSE).
