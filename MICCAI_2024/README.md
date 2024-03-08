# HeartResearch

# Abstract 
<p align="justify">In the field of Electrocardiogram (ECG) classification, two primary obstacles arise. Firstly, existing ECG datasets consistently demonstrate imbalances and biases across various modalities. Secondly, the time-series data comprises diverse lead signals, causing Convolutional Neural Networks (CNNs) to become overfitting to high-power ones, hence diminishing the learning efficiency of the Deep Learning (DL) process. Additionally, when the ECG signal is short, performance from such high-dimensional data may be susceptible to overfitting. Despite these evident challenges, current efforts predominantly focus on enhancing DL models by designing novel architectures, seemingly overlooking the core issues. This narrow focus hinders advancements in ECG classification. To tackle these challenges, our proposed method introduces two simple and direct techniques for improving ECG classification learning. To address the high dimensionality issue, we employ an Inverted Channel-wise Attention Squeeze and Excitation (ICWA-SE) on signal-encoded images. This approach reduces redundancy in the feature data range, highlighting changes in the dataset. Simultaneously, to counteract data imbalance, we propose Inverted Weight Logarithmic Loss (IWL) to alleviate imbalances among the data. Our quantitative experiments indicate significantly faster convergence and higher accuracy in ECG classification compared to other baselines from $2\%$ to $7\%$. 

# Folder structure
The zip file that we provide includes 3 folder:
- Folder "Proposed method": includes 
# Experiment
## Setup
This work can be conducted on any platform: Windows, Ubuntu, Google Colab. In Windows or Ubuntu use the following script to create a virtual environment.
```
git clone https://github.com/ScaleMind-C9308A/HeartResearch.git
cd path/to/HeartResearch
python -m venv .env
```
The Python packages used in this project are listed below. All the packages can be installed by command ```pip install -r requirements.txt```.
```
wheel 
pandas==1.4.4
matplotlib==3.6.0
numpy==1.19.5
tqdm==4.65.0
jupyter
torch==2.0.1
torchaudio==2.0.2
torchvision==0.15.2
torchmetrics==0.11.4
scipy==1.11.1
scikit-learn==1.3.0
datetime

```
[Pytorch](https://pytorch.org/) is the main package for conducting optimization calculations.

## Running
Directly run the jupyter notebook file "  " and easily reproduce the results.
