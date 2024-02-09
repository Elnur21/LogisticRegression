Introduction:
This paper discusses the perfrmance of DNNs in TSC tasks. 
Given the need to accurately classify time series data, researchers have proposed hundreds of methods to solve this task. One of the most popular and traditional TSC approaches is the use of a nearest neighbor (NN) classifier coupled with a distance function. 
The Dynamic Time Warping (DTW) distance when used with a NN classifier has been shown to be a very strong baseline

This paper targets the following open questions:
What is the current state-of-the-art DNN for TSC ?
Is there a current DNN approach that reaches state-of-the-art performance for TSC and is less complex than HIVE-COTE ? What type of DNN architectures works best for the TSC task ? How does the random initialization affect the performance of deep learning classifiers?
Could the black-box effect of DNNs be avoided to provide interpretability?


Background:
Although there exist many types of DNNs, in this review 3 main architectures are given for theoretical background on training DNNs for the TSC task: MLP, CNN, ESN.
MLP: in 1 layer each neuron is connected to all neurons in previous layer, and learns directly from raw input, but may lose temporal information. 
CNN: each neuron in a layer is connected to a local receptive field in the previous layer through convolutional operations. 
ESN: in 1 layer each neuron is connected to all neurons in previous layer but fully connected.

Deep learning approaches for TSC can be separated into two main categories: the generative and the discriminative models.

Generative models -> (Auto Encoders, Echo State Networks)
Auto Encoders -> (SDAE, CNN, DBN, RNN)
Echo State Networks -> (traditional, kernel learning, meta learning)
Generative models usually exhibit an unsupervised training step that precedes the learning phase of the classifier.
This type of networks refers to as Model-based classifiers.
Auto Encoders: to reconstruct input data, capturing features in a latent space. Variants include SDAE, CNN, and DBN.
Echo State Networks: input data through a fixed reservoir of neurons:
Traditional ESNs for feature extraction.
Kernel Learning ESNs for classification.
Meta-learning ESNs for optimizing architectures.

Discriminative models -> (Feature Engineering, End-to-End)
Feature Engineering -> (image transform, domain specific)  
End-to-End -> (MLP, CNN, Hybrid)
A discriminative model that directly learns the mapping between the raw input of a time series and outputs a probability distribution over the class variables in a dataset.
Feature Engineering: This involves transforming time series into images using methods like Gramian fields, recurrence plots, and Markov transition fields. It also includes extracting domain-specific features, such as velocity in surgical training tasks.
End-to-End: These models incorporate feature learning while fine-tuning the classifier.
Hybrid: Combines CNN with other architectures like Gated Recurrent Units and attention mechanisms, showing promising results.
And in this review the discriminative models are selected.


Approaches:
Here the nine architectures are presented such as MLP, FCN, ResNet, MCNN, t-LeNet, MCDCNN, Time-CNN, TWIESN, Encoder.
-- MLP: The network contains 4 layers in total where each one is fully connected to the output of its previous layer.
-- FCNs: do not contain any local pooling layers which means that the length of a time series is kept unchanged throughout the convolutions. The replacement of the traditional final FC layer with a Global Average Pooling (GAP) layer reduces drastically the number of parameters in a neural network while enabling the use of the CAM. 
-- ResNet: The network is composed of three residual blocks followed by a GAP layer and a final softmax classifier whose number of neurons is equal to the number of classes in a dataset.
-- Encoder is like FCN but it uses attention layer instead of global average pooling.
-- MCNN is characterized by a traditional CNN structure and augmented with the Window Slicing method for data processing.
-- t-LeNet consists of two convolutions followed by an FC layer and a softmax classifier.
-- MCDCNN is used for multivariant time series. It employs a traditional deep CNN architecture with parallel convolutions applied independently on each dimension of the input MTS, enhancing its effectiveness for this type of data.
-- Time-CNN offers unique features for both univariate and multivariate time series classification. For all the categorical cross entropy is used exclude Time-CNN because Mean Square Error is used for it as loss function. 
-- TWIESN: It is s the only non-convolutional recurrent architecture. It employs Ridge classifiers to predict class probabilities for each time series element.
The best model on the validation set or training set loss is chosen for evaluation. For FCN, ResNet, and MLP, the learning rate was reduced when the training loss did not improve for 50 consecutive epochs.


Experimental setup:
-- Univariate archive:
Each algorithm was tested on the whole UCR/UEA archive which contains 85 univariate time series datasets. 
The datasets possess different varying characteristics such as the length of the series which has a minimum value of 24 for the ItalyPowerDemand dataset and a maximum equal to 2,709 for the HandOutLines dataset. 
Note that the time series in this archive are already z-normalized
-- Multivariate archive:
Baydogan’s archive contains 13 MTS classification datasets. This archive also exhibits datasets with different characteristics such as the length of the time series which, unlike the UCR/UEA archive, varies among the same dataset.
-- Experiments:
8730 models were trained on 97 datasets (12 datasets are multivariate). 
The experimental setup involved training nine deep learning models on a cluster of more than 60 GPUs. Each dataset was trained for 10 different runs. 
The mean accuracy over the 10 runs was taken to reduce bias due to weight initialization. 
The results showed that deep learning models were able to significantly outperform the NN-DTW model and achieve similar results to COTE and HIVE-COTE using a deep residual network architecture.


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
