# Unsupervised and Ensemble-based Anomaly Detection Method for Network Security
This project is for the paper "Unsupervised and Ensemble-based Anomaly Detection Method for Network Security" in *The 2022-14th International Conference on Knowledge and Smart Technology*.
- Authors of source code: yangdonghun3@kisti.re.kr
- Current version of the project: ver. 0.1

## Abstract
Bigdata and IoT technologies are developing rapidly. Accordingly, consideration of network security is also emphasized, and efficient intrusion detection technology is required for detecting increasingly sophisticated network attacks. In this study, we propose an efficient network anomaly detection method based on ensemble and unsupervised learning. The proposed model is built by training an autoencoder, a representative unsupervised deep learning model, using only normal network traffic data. The anomaly score of the detection target data is derived by ensemble the reconstruction loss and the Mahalanobis distances for each layer output of the trained autoencoder. By applying a threshold to this score, network anomaly traffic can be efficiently detected. To evaluate the proposed model, we applied our method to UNSW-NB15 dataset. The results show that the overall performance of the proposed method is superior to those of the model using only the reconstruction loss of the autoencoder and the model applying the Mahalanobis distance to the raw data.

## Requirements
- Python 3.7
- Pytorch 1.5 (GPU version is recomended)
- scipy
- scikit-learn
- numpy
- pandas

## Contact information
- Donghun Yang. yangdonghun3@kisti.re.kr
- Myunggwon Hwang. mgh@kisti.re.kr
