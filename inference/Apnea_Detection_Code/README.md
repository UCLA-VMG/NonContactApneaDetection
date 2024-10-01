# Sleep Apnea Detection using Envelope Detection

This folder contains the implementation of an envelope detection-based algorithm for identifying sleep apnea events from thermal and radar data. This algorithm extracts the breathing waveform from thermal imaging and radar signals, and detects anomalies corresponding to apnea events and the breathing rate of the patient.

## Overview

This repository implements a non-contact method for sleep apnea detection based on two sensor modalities:
1. **Thermal Imaging**: Measures oro-nasal airflow using thermal cameras.
2. **Radar**: Captures chest movements (respiratory effort) using FMCW radar.

The envelope detection algorithm processes the breathing signal from these modalities to detect apneas. By analyzing the upper and lower envelopes of the signal, it identifies periods of breathing interruption (apnea) and classifies them accordingly.
