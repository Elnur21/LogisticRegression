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
