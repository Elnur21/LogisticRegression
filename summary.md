Introduction:
This paper discusses the perfrmance of DNNs in TSC tasks. 
Background:
Although there exist many types of DNNs, in this review 3 main architectures are given for theoretical background on training DNNs for the TSC task: MLP, CNN, ESN.
MLP: in 1 layer each neuron is connected to all neurons in previous layer. 
CNN: each neuron in a layer is connected to a local receptive field in the previous layer through convolutional operations. 
ESN: : in 1 layer each neuron is connected to all neurons in previous layer but fully connected.
Deep learning approaches for TSC can be separated into two main categories: the generative and the discriminative models. And in this review the discriminative models are selected.
Approaches:
Here the nine architectures are presented such as MLP, FCN, ResNet, MCNN, t-LeNet, MCDCNN, Time-CNN, TWIESN, Encoder.
Encoder is like FCN but it uses attention layer instead of global average pooling.
MCNN is characterized by a traditional CNN structure and augmented with the Window Slicing method for data processing.
t-LeNet consists of two convolutions followed by an FC layer and a softmax classifier.
MCDCNN is used for multivarian time series.
For all the categorical cross entropy is used exclude Time-CNN because Mean Square Error is used for it as loss function. 
The best model on the validation set or training set loss is chosen for evaluation. For FCN, ResNet, and MLP, the learning rate was reduced when the training loss did not improve for 50 consecutive epochs.
Datasets:
8730 models were trained on 97 datasets (12 datasets are multivariate). The experimental setup involved training nine deep learning models on a cluster of more than 60 GPUs. Each dataset was trained for 10 different runs. The mean accuracy over the 10 runs was taken to reduce bias due to weight initialization. The results showed that deep learning models were able to significantly outperform the NN-DTW model and achieve similar results to COTE and HIVE-COTE using a deep residual network architecture.
Visualisation:
For visualization CAM is used on GunPoint and Meat datasets to reduce the black-box effect. And MDS is used to gain insights into the spatial distribution of input time series belonging to different classes in the dataset.





InceptionTime
Introduction:
The paper introduces InceptionTime, a novel deep learning ensemble for TSC, achieving state-of-the-art accuracy and providing insights into its success through an analysis of architectural hyperparameters.
Background:
The state-of-the-art for TSC has been organized into the following main categories: Whole series, Dictionary-based, Shapelets, and Transformation ensembles. 
Whole series compares two series using a certain distance.
Dictionary-based focuses on discriminating time series by the frequency of repetition of specific sub-series.
Shapelets focuses on finding relatively short repeated subsequences to identify a certain class.
Transformation ensembles in TSC refer to the combination of various techniques for transforming time series data, aiming to improve accuracy by leveraging diverse representations of the data through ensemble methods.
InceptionTime:
InceptionTime model includes an ensemble of 5 Inception networks, each comprising two residual blocks, three Inception modules, and a Global Average Pooling layer. 
The Inception module is a key component of the InceptionTime architecture for time series classification. It is inspired by the Inception-v4 architecture and consists of multiple filters applied simultaneously to an input time series. And the Inception module involves a "bottleneck" layer for dimensionality reduction and sliding multiple filters of different lengths simultaneously.
The bottleneck layer performs an operation of sliding filters of length 1 on the input multivariate time series (MTS) with a stride of 1.
Conclusion:
InceptionTime is an ensemble of Inception-based networks that achieves state-of-the-art results with significant scalability and speed, highlighting the potential for applying these advancements to multivariate TSC.
