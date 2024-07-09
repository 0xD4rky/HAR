# Human Activity Recognition-
The aim of the model is to detect various common activities peformed by humans and classify these activities to their respective classes.
this helps in improving the Robot-Human Interaction and also in Anomaly Detection.

### You can download the dataset For HAR from following link-
https://www.kaggle.com/datasets/pevogam/ucf101/download?datasetVersionNumber=1

## Dataset
UCF-101 dataset consists of 101 classes of various human activities and each class contains several video clips of very short length.
The UCF-101 dataset consists of a total of 13,320 videos, distributed across the 101 action categories.

## Libraries Used
#### Numpy 
#### Pandas
#### Transformers
#### Pytorch
#### cv2
#### Tensorflow
#### Matplotlib

## Model Architecture
The pipeline of the model follows the Base-Deit-16 Model which is connected to Sequential Layer containing: <br> 
Implemented Vision Transformer with TSM Module
1. Linear Layer which decreases the number of parameters to work with to 512<br>
2. Relu Activation Function
3. Dropout of 0.3 for Regularization
4. Final Output Layer which outputs the probabilities of input to belong 101-Classes.

## HyperParameters:
1. Learning Rate of 0.0005.
2. Learning Rate Scheduler which decreases the learning rate by the factor of 0.97 after every 3 epochs.
3. Criterion Used is LabelSmoothingCross Entropy

## References
https://huggingface.co/docs/transformers/model_doc/vit<br>
https://www.youtube.com/watch?v=Ssndsjh1Zqk<br>
https://arxiv.org/pdf/2108.02432.pdf<br>
https://arxiv.org/pdf/1611.05267.pdf<br>
https://arxiv.org/pdf/2204.00452.pdf/<br>
https://arxiv.org/pdf/1611.05267.pdf<br>
