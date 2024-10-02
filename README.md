# Thermal Imaging and Radar for Remote Sleep Monitoring of Breathing and Apnea

This repository contains the code used in the paper **"Thermal Imaging and Radar for Remote Sleep Monitoring of Breathing and Apnea"**, authored by Kai Del Regno, Alexander Vilesov, Adnan Armouti, Anirudh Bindiganavale Harish, Selim Emir Can, Ashley Kita, and Achuta Kadambi.

## Abstract

Polysomnography (PSG) is the gold standard for diagnosing sleep disorders, but it is cumbersome and expensive. At-home sleep apnea testing (HSAT) solutions are contact-based, which may not be tolerated by some patient populations. This paper presents the first comparison of radar and thermal imaging for non-contact sleep monitoring. Our results show that thermal imaging outperforms radar in detecting apneas. Additionally, we propose a multimodal method to classify obstructive and central sleep apneas using synchronized radar and thermal data.

- Thermal method: Accuracy 0.99, Precision 0.68, Recall 0.74
- Radar method: Accuracy 0.83, Precision 0.13, Recall 0.86

We also provide code, along with the hardware setup, and circuit schematics for synchronized radar and thermal data collection.

## Repository Contents

1. data_collection, contains all code, circuit diagrams and instructions for running code related to capturing data from the thermal and radar sensors. 
2. preprocessing, after data-collection, the data requires pre-processing before being sent to the inference stage. The associated instructions for using our preprocessing library are located in the README in "preprocessing/mmProc"
3. inference/Apnea_Detection_Code, contains the scripts "apnea_detection.py" and "breathing_detection.py" and accompanying code for running breathing and apnea estimation. 

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{delregno2024thermal,
  title={Thermal Imaging and Radar for Remote Sleep Monitoring of Breathing and Apnea},
  author={Del Regno, Kai and Vilesov, Alexander and Armouti, Adnan and Harish, Anirudh Bindiganavale and Can, Selim Emir and Kita, Ashley and Kadambi, Achuta},
  journal={arXiv preprint arXiv:2407.11936v2},
  year={2024}
}
