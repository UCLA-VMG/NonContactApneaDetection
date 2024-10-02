# Thermal Imaging and Radar for Remote Sleep Monitoring of Breathing and Apnea

This repository contains the code used in the paper **"Thermal Imaging and Radar for Remote Sleep Monitoring of Breathing and Apnea"**, authored by [Kai Del Regno](https://www.linkedin.com/in/kai-del-regno-103269222/), [Alexander Vilesov](https://asvilesov.github.io/), [Adnan Armouti](https://adnan-armouti.github.io/), [Anirudh Bindiganavale Harish](https://anirudh0707.github.io/), [Selim Emir Can](https://selim-emir-can.github.io/), [Ashley Kita](https://www.uclahealth.org/providers/ashley-kita), and [Achuta Kadambi](https://www.ee.ucla.edu/achuta-kadambi/).

## Abstract

Polysomnography (PSG) is the gold standard for diagnosing sleep disorders, but it is cumbersome and expensive. At-home sleep apnea testing (HSAT) solutions are contact-based, which may not be tolerated by some patient populations. This paper presents the first comparison of radar and thermal imaging for non-contact sleep monitoring. Our results show that thermal imaging outperforms radar in detecting apneas. Additionally, we propose a multimodal method to classify obstructive and central sleep apneas using synchronized radar and thermal data.

- **Thermal method**: Accuracy 0.99, Precision 0.68, Recall 0.74
- **Radar method**: Accuracy 0.83, Precision 0.13, Recall 0.86

We also provide the code, hardware setup, and circuit schematics for synchronized radar and thermal data collection.

## Contributions

1. **A comparison of radar and thermal modalities** for remote sleep monitoring and detection of breathing and sleep apnea.
2. **A non-contact multimodal thermal and radar stack** for the detection and clinically relevant classification of sleep apnea.
3. **A dataset composed of 10 sleeping patients** with hardware-synchronized thermal videos, frequency-modulated continuous wave (FMCW) radar data, and ground truth waveforms and annotations by a certified sleep technician at a sleep center. We also open-source our data-collection framework, code base, and circuit schematics for collecting hardware-synchronized radar and thermal data.

## Repository Contents

1. **data_collection/**: Contains all code, circuit diagrams, and instructions for capturing data from the thermal and radar sensors.
2. **preprocessing/**: After data collection, the data requires preprocessing before moving to the inference stage. The associated instructions for using our preprocessing library are located in the README within "preprocessing/mmProc".
3. **inference/Apnea_Detection_Code/**: Contains the scripts `apnea_detection.py` and `breathing_detection.py`, along with accompanying code for running breathing and apnea estimation.

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{delregno2024thermal,
  title={Thermal Imaging and Radar for Remote Sleep Monitoring of Breathing and Apnea},
  author={Del Regno, Kai and Vilesov, Alexander and Armouti, Adnan and Harish, Anirudh Bindiganavale and Can, Selim Emir and Kita, Ashley and Kadambi, Achuta},
  journal={arXiv preprint arXiv:2407.11936v2},
  year={2024}
}
