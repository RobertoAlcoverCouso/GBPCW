Code based on [HRDA](https://github.com/lhoyer/HRDA). 
## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/hrda
source ~/venv/hrda/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
conda install -c conda-forge cvxopt # package for optimal weight computing
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

Create the directory to dump logs and results:
```shell
mkdir work_dirs
```

## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

The final folder structure should look like this:

```none
├── HRDA
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── GTAV
│   │   ├── images
│   │   ├── labels
│   ├── synthia (optional)
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ├── ...
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the sampling indexes:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Testing & Predictions

The provided GBPCW checkpoint trained on [GTA→Cityscapes](link/to/weights) can be tested on the Cityscapes validation set using:

```shell
sh test.sh work_dirs/gtaHR2csHR_hrda_246ef
```
The mIoU of the model is printed to the console.
The provided checkpoint should achieve 74.7 mIoU. 


## Training

For convenience, we provide an [annotated config file](configs/hrda/gtaHR2csHR_hrda.py)
to replicate training on GBPCW. A training job can be launched using:

```shell
python run_experiments.py --config configs/hrda/gtaHR2csHR_hrda.py
```

The logs and checkpoints are stored in `work_dirs/`.

For the other experiments in our paper, we use the following scripts:

```none
├── HRDA
├── ├── configs
├── ├── ├── hrda
├── ├── ├── ├── gtaHR2csHR_hrda.py
├── ├── ├── ├── synHR2csHR_hrda.py
├── ├── ├── ├── ...
├── ├── ├── daformer
├── ├── ├── ├── gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0
├── ├── ├── ├── syn2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0
├── ├── ├── ├── ...
├── ├── ├── ...
├── ├── ...
├── ...
```
When evaluating a model trained on Synthia→Cityscapes, please note that the
evaluation script calculates the mIoU for all 19 Cityscapes classes. However,
Synthia contains only labels for 16 of these classes. Therefore, it is a common
practice in UDA to report the mIoU for Synthia→Cityscapes only on these 16
classes. As the Iou for the 3 missing classes is 0, you can do the conversion
`mIoU16 = mIoU19 * 19 / 16`.

## Checkpoints

Below, we provide checkpoints for different architectures.

* [HRDA for GTA→Cityscapes]()
* [HRDA for Synthia→Cityscapes]()
* [DAFormer for GTA→Cityscapes]()
* [DAFormer for Synthia→Cityscapes]()

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU for 19 classes. For Synthia→Cityscapes, it is
  necessary to convert the mIoU to the 16 valid classes. Please, read the
  section above for converting the mIoU.

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files are:

* [configs/hrda/gtaHR2csHR_hrda.py](configs/hrda/gtaHR2csHR_hrda.py):
  Annotated config file for the final HRDA.
* [mmseg/models/losses/cross_entropy_loss.py](mmseg/models/losses/cross_entropy_loss.py):
  Implementation of the Gradient Based per-class loss.
* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of the DACS UDA framework.

## Acknowledgements

GBPCW is based on the following open-source projects. We thank their
authors for making the source code publicly available.
* [HRDA](https://github.com/lhoyer/HRDA)
* [LOW](https://github.com/cajosantiago/LOW)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)

## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) and [HRDA](https://github.com/lhoyer/HRDA/LICENSES.md) for the careful check, 
if you are using our code for commercial matters.
