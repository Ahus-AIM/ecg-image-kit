# `ECG-Image-Kit`
***A toolkit for synthesis, analysis, and digitization of electrocardiogram images***

## Citation
Please include references to the following articles in any publications:

1. Kshama Kodthalu Shivashankara, Deepanshi, Afagh Mehri Shervedani, Matthew A. Reyna, Gari D. Clifford, Reza Sameni (2024). ECG-image-kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. In Physiological Measurement. IOP Publishing. doi: [10.1088/1361-6579/ad4954](https://doi.org/10.1088/1361-6579/ad4954)


2. ECG-Image-Kit: A Toolkit for Synthesis, Analysis, and Digitization of Electrocardiogram Images, (2024). URL: [https://github.com/alphanumericslab/ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit)

## Contributors
- Elias Stenhede, Department of Medical Technology and E-health, Akershus University Hospital, Norway
- Agnar Martin Bjørnstad, Department of Medical Technology and E-health, Akershus University Hospital, Norway
- Deepanshi, Department of Biomedical Informatics, Emory University, GA, US
- Kshama Kodthalu Shivashankara, School of Electrical and Computer Engineering, Georgia Institute of Technology, Atlanta, GA, US
- Matthew A Reyna, Department of Biomedical Informatics, Emory University, GA, US
- Gari D Clifford, Department of Biomedical Informatics and Biomedical Engineering, Emory University and Georgia Tech, GA, US
- Reza Sameni (contact person), Department of Biomedical Informatics and Biomedical Engineering, Emory University and Georgia Tech, GA, US

## Installation
- Clone this repository and move to the root folder of the repository.
- Create a venv with python3.12:
     ```
     python3.12 -m venv venv
     ```
- Activate the venv:
     ```
     source venv/bin/activate
     ```
- Install the required packages:
     ```
     python3 -m pip install -r requirements.txt
     ```

## ECG Data

A few samples from the PTB-XL  dataset are already stored in [data](./data/ptb-xl/00000). If you with to generate a diverse dataset, you should download the full dataset. If you use the examples or the full dataset, please cite:

1. Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. doi: [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)

## Run the pipeline
1. Take a look in the [config file](./src/config/config.yml)
2. Run the pipeline with the following command:
     ```
     python src/generate_images.py
     ```

![Static Badge](https://img.shields.io/badge/ecg_image-kit-blue)