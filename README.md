# MRAtoBG-brain-vessel-segmentation COSTA package

## Setup
### 1. **Requirements**

To successfully run the COSTA framework for MRAtoBG-brain-vessel-segmentation, please ensure the following requirements are met:

<center>Ubuntu 20.04 LTS + NVIDIA RTX 3090 + CUDA version 12.0</center>

### 2. **Requirements Installation**

To install the necessary components, please follow the steps below:

- Create a new Python 3.9 environment named MRAtoBG-brain-vessel-segmentation using Conda:

  ```bash
  conda create -n MRAtoBG-brain-vessel-segmentation python=3.9 # Python 3.8 or Python 3.10 is also acceptable.
  ```

- Activate the MRAtoBG-brain-vessel-segmentation environment:

  ```bash
  conda activate MRAtoBG-brain-vessel-segmentation
  ```

- Clone the MRAtoBG-brain-vessel-segmentation COSTA repository from GitHub:

  ```bash
  git clone https://github.com/jshe690/COSTA.git
  ```

- Navigate to the COSTA directory:

  ```bash
  cd ./COSTA
  ```

- Install the required dependencies:

  ```shell
  pip install -e .
  ```
