## G-RNN and R-GNN 
### Optimizing Spatio-Temporal Reasoning in Neural Networks

Combining Graph Neural Networks (GNNs) with Recurrent Neural Networks (RNNs) offers a promising approach to enhance spatio-temporal reasoning in diverse applications, from autonomous driving to behavior analysis in areas such as security, medical assistive devices, and traffic management. However, a definitive agreement on the best sequence or combination of these networks has yet to be established. This paper presents two innovative layer architectures that merge GNNs and RNNs more seamlessly than the traditional sequential arrangements. Specifically, we investigate the potential of embedding one network within the other, resulting in a more integrated representation of spatial and temporal features. Through experiments on predicting supply deficits using graph-based time-series forecasting, we demonstrate that these merged architectures outperform traditional models. The RGNN architecture, in particular, stood out as the most effective approach, underscoring the potential of architectural fusions in tackling intricate spatio-temporal challenges.

______________________________________________________________________
To set-up your environment, run:  
```
$ conda create --name <env> --file requirements.txt
```