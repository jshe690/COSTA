# COSTA package for MRAtoBG_brain_vessel_segmentation
---

## ⚠️ System Requirements

Please ensure your system meets these requirements for full compatibility:

- **OS**: Ubuntu 20.04
- **Python**: 3.8
- **GPU**: NVIDIA V100
- **CUDA**: Version 11.7

---

## **Requirements Installation**

Before continuing, ensure you have completed step `1. Clone the Repository` from [here](https://github.com/ABI-Animus-Laboratory/MRAtoBG_brain_vessel_segmentation).

You should now have a system variable to the MRAtoBG_brain_vessel_segmentation repo, for example:

```bash
export MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH=/user/repos/MRAtoBG_brain_vessel_segmentation
```

To install the necessary COSTA components, please follow these steps:

- Create a new Python 3.8 environment:

  ```bash
  python -m venv $MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv
  ```

- Activate the venv:

  ```bash
  source $MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/bin/activate
  ```

- Navigate to the parent directory of `$MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH` (e.g., `/user/repos`), and clone this repo, for example:

  ```bash
  cd $MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH
  cd ..
  git clone https://github.com/jshe690/COSTA.git
  ```

- Navigate to the COSTA directory [COSTA_DIRECTORY]:

  ```bash
  cd ./COSTA
  ```

- Install the required dependencies:

  ```shell
  pip install -e .
  ```

- Manually replace nnUNet's `segmentation_export.py` file:

Rename:
`$MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/lib/python3.8/site-packages/nnunet/inference/segmentation_export.py`

To:
`$MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/lib/python3.8/site-packages/nnunet/inference/segmentation_export_orig.py`

And copy:
`[COSTA_DIRECTORY]/extras/segmentation_export.py`

To:
`$MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/lib/python3.8/site-packages/nnunet/inference`


- Manually replace nnUNet's `neural_network.py` file:

Rename:
`$MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/lib/python3.8/site-packages/nnunet/network_architecture/neural_network.py`

To:
`$MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/lib/python3.8/site-packages/nnunet/network_architecture/neural_network_orig.py`

And copy:
`[COSTA_DIRECTORY]/extras/neural_network.py`

To:
`$MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/lib/python3.8/site-packages/nnunet/network_architecture`
