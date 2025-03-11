# Reconstructing ECG from Indirect Signals: A Denoising Diffusion Approach

## Datasets
* [ptbxl](https://physionet.org/content/ptb-xl/1.0.3/): ```python preprocessing/generate_ptb_xl.py <path of download ptbxl dataset> <target sampling frequency>```
  * description of available statements [scp_statements](https://physionet.org/content/ptb-xl/1.0.1/scp_statements.csv) (scp_statements)
* download noise database [nstdb](https://physionet.org/content/nstdb/1.0.0/)

## Diffusion Generative Prior for 12-lead ECGs
* train: ```python diffusion_prior/train.py model.unconditional=false```
* generate: ```python diffusion_prior/generate.py model.unconditional=false dataset.training_class=test```
* evaluation:
  * classifier trained on real data: ```python diffusion_prior/downstream_classifier_ptbxlStrodoff.py evaluate.train_downstream='real'```
  * classifier trained on synthetic data (after generated samples with `python diffusion_prior/generate.py`): ```python diffusion_prior/downstream_classifier_ptbxlStrodoff.py evaluate.train_downstream='gen'```

## Missing-Block
* Midpoint-Guidance Posterior Sampling algoirthm (MGPS): ```python inpainting_ecg.py algo.missingness_type=bm algo.missingness=50 train.results_path=<path of the prior, trained on 2.5 seconds for comparizon with baseline>```

## 12-leads reconstruction from single-lead I
* MGPS: ```python inpainting_ecg.py algo.missingness_type=lead algo.missingness=1 train.results_path=<path of the prior>```
* eval: ```python eval_inpainting_ecg.py algo.missingness_type=lead algo.missingness=1 train.results_path=<path of the prior>```

## Denoising on MIT-BIH Noise database
* Baseline Wander: ```python denoising_mpgs.py denoising.noise_type='bw' train.results_path=<path of the prior>```
* Electrode Motion: ```python denoising_mpgs.py denoising.noise_type='em' train.results_path=<path of the prior>```
