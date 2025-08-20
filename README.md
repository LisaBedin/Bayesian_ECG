# Reconstructing ECG from Indirect Signals: A Denoising Diffusion Approach
This repository provides tools to reconstruct 12-lead ECGs from indirect signals (e.g. incomplete and / or noisy ECGs) using a denoising diffusion approach. You can skip the preprocessing and training steps by using the preprocessed datasets and pretrained models provided below.

## Quick Start (Using Pretrained Models)
| Resource | Description | Link |
|----------|-------------|------|
| **Preprocessed ECGs** | Validation and test sets (10s ECGs at 100Hz). Each patient is stored as a .npz file containing: <br> - **ecg**: 12-lead ECG signal of shape 12 × 1000 (12 leads, 10 seconds at 100Hz). <br> - **all_labels**: List of diagnostic labels (e.g., ["SB", "NORM"]). | [Download](https://drive.google.com/drive/folders/1R4WPrJOZ6M3EC5_46T6GQoNNLKbbwqZ1?usp=share_link) |
| **Pretrained Diffusion Prior** | Pretrained model for 10s ECGs, trained conditionally on 71 SCP statements (cardiac disease diagnoses) | [Download](https://drive.google.com/drive/folders/1CyZvVza4q5SHlTkUlexumiLgb2vz4q6t?usp=share_link) |

## Datasets Preprocessing
### PTB-XL Dataset (Optional)
* Download: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
* Preprocessing: Run the following script to preprocess the dataset (optional if using preprocessed data):
  ```python preprocessing/generate_ptb_xl.py <path of download ptbxl dataset> <target sampling frequency>```
  > **Note**: In our experiments, we used a sampling frequency of 100Hz for faster processing. You can adjust this parameter based on your needs.
* SCP Statements: Description of available diagnostic statements: description of available statements [scp_statements](https://physionet.org/content/ptb-xl/1.0.1/scp_statements.csv)
### Noise Database
* Download: [NSTDB 1.0.0](https://physionet.org/content/nstdb/1.0.0/)

## Diffusion Generative Prior for 12-lead ECGs (Optional)
### Training
If you want to train the model from scratch conditionally on SCP statements, set model.unconditional=false to enable training with diagnostic labels:
```python diffusion_prior/train.py model.unconditional=false```
### Generation
Generate synthetic ECGs using the pretrained model
* generate: ```python diffusion_prior/generate.py model.unconditional=false dataset.training_class=test```
* evaluation:
  * classifier trained on real data: ```python diffusion_prior/downstream_classifier_ptbxlStrodoff.py evaluate.train_downstream='real'```
  * classifier trained on synthetic data (after generated samples with `python diffusion_prior/generate.py`): ```python diffusion_prior/downstream_classifier_ptbxlStrodoff.py evaluate.train_downstream='gen'```

## 12-leads reconstruction from single-lead I
* MGPS: ```python inpainting_ecg.py algo.missingness_type=lead algo.missingness=1 train.results_path=<path of the prior>```
* eval: ```python eval_inpainting_ecg.py algo.missingness_type=lead algo.missingness=1 train.results_path=<path of the prior>```

## Denoising on MIT-BIH Noise database
* Baseline Wander: ```python denoising_mpgs.py denoising.noise_type='bw' train.results_path=<path of the prior>```
* Electrode Motion: ```python denoising_mpgs.py denoising.noise_type='em' train.results_path=<path of the prior>```

## Optional: Missing-Block (For Research Comparison)
This section is only relevant if you want to compare results with existing literature. It requires training a specific model on 2.5-second segments, which is not the default setup.
```python inpainting_ecg.py algo.missingness_type=bm algo.missingness=50 train.results_path=<path_to_specific_prior>```
> **Note**: The prior should be trained on 2.5-second segments for comparison with baselines.

## Citation
If you use this code, please cite the following
```
@article{bedin2025reconstructing,
  author  = {Bedin, Lisa and Janati, Yazid and Victorino Cardoso, Gabriel and Duchateau, Josselin and Dubois, R{\'e}mi and Moulines, Eric},
  title   = {Reconstructing {ECG} from indirect signals: a denoising diffusion approach},
  journal = {Phil. Trans. R. Soc. A},
  volume  = {383},
  year    = {2025},
  pages   = {20240330},
  doi     = {10.1098/rsta.2024.0330},
  url     = {https://doi.org/10.1098/rsta.2024.0330}
}
```
