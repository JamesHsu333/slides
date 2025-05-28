# GAMIVAL-IQA: Image Quality Assessment Using GAMIVAL Pipeline

This repository contains a modified version of the [GAMIVAL](https://github.com/utlive/GAMIVAL) pipeline, originally designed for no-reference video quality assessment (VQA), adapted here for no-reference image quality assessment (IQA).

The original GAMIVAL was proposed in IEEE SPL 2023, achieving top performance on gaming content. This adaptation reuses its architecture and SVR-based regression framework to evaluate still images, especially tailored to datasets such as Cinnamoroll-themed image collections.

## Acknowledgment
This codebase is based on the original work by Chen et al., “GAMIVAL: Video Quality Prediction on Mobile Cloud Gaming Content” (IEEE SPL 2023). Please cite the original paper if you use any part of this pipeline in your work.


## Modifications Summary
- Adapted the CNN and feature extraction pipeline for still images instead of video frames.
- Renamed and revised the evaluation, training, and testing scripts:
    - demo_compute_CNN_feats_IQA.py
    - evaluate_bvqa_features_regression_IQA.py
    - train_SVR_IQA.py
    - test_SVR_IQA.py
- Modified data input/output handling to support typical IQA datasets with Mean Opinion Score (MOS) labels for images.
- Dependencies trimmed and aligned for macOS. See requirements_for_mac.txt.

## Requirements

Install the dependencies (macOS):

```bash
pip install -r requirements_for_mac.txt
```

## IQA Demos

### CNN Feature Extraction (Image)

```bash
$ python demo_compute_CNN_feats_IQA.py \  
--image_dir path_to_images \
--model ./models/subjectiveDemo2_DMOS_Final.h5 \
--csv path_to_mos_csv \
--save_path feat_files/{mat_file_name} \
```

### Evaluation of IQA Model

```bash
$ python evaluate_bvqa_features_regression_IQA.py \
--feature_file feat_files/{mat_file_name} \
--out_file {evaluate_file_name}
```

### Training a SVR Model

```bash
$ python train_SVR_IQA.py \
--feature_file feat_files/{mat_file_name} \
--best_parameter best_param/{best_param_file_name} \
--train_csv path_to_mos_train_csv
```

### Predicting MOS (Testing)

```bash
$ python test_SVR_IQA.py \
--feature_file feat_files/{mat_file_name} \
--best_parameter best_param/{best_param_file_name} \
--test_csv path_to_mos_test_csv
```

## Citation

If you use this project (original or modified), please cite the original authors:

[Y.-C. Chen, A. Saha, C. Davis, B. Qui, X. Wang, I. Katsavounidis, and A. C. Bovik, “Gamival : Video quality prediction on mobile cloud gaming content,” *IEEE Signal Processing Letters*, 2023, doi: 10.1109/LSP.2023.3255011.](https://doi.org/10.1109/LSP.2023.3255011)

```
@ARTICLE{10065464,
  author={Chen, Yu-Chih and Saha, Avinab and Davis, Chase and Qiu, Bo and Wang, Xiaoming and Gowda, Rahul and Katsavounidis, Ioannis and Bovik, Alan C.},
  journal={IEEE Signal Processing Letters}, 
  title={GAMIVAL: Video Quality Prediction on Mobile Cloud Gaming Content}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/LSP.2023.3255011}}
```


## Contact

For original GAMIVAL questions:
- Yu-Chih Chen (berriechen@utexas.edu)
- Avinab Saha (avinab.saha@utexas.edu)
- Prof. Alan C. Bovik (bovik@ece.utexas.edu)

For this IQA adaptation:
Please refer to this repository or open an issue for inquiries.