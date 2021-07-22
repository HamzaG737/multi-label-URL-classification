# Multi-label classification for website URL
In this repo , we build a multi-label classifier that takes as input a web url and returns a list of categories for the URL. In this project we don't scrape the web pages and hence use only the information provided by the url itself. 

# Usage 
First install the necessary packages using the following command : 
```shell
$ pip install -r requirements.txt
```
### Infering categories
Given a text file where each line corresponds to an url , you can infer the categories using the following command : 
```shell
$ python infer_categories.py --path_to_input path/to/text_file
```
You will need two binary models for the inference , a fasttext model and the classifier one ( MLARAM , see methods for more details). You can download them via this [link](https://drive.google.com/drive/folders/1ZZnT8zSFFbkF2nhyfXGkeX5-JrGzy8-M?usp=sharing) . After download , you need to store them in a models directory that shares the same workspace as infer_categories.py . If you want to avoid these steps , you can train from scratch ( see next subsection) and then run the inference module. 

The results will be stored in a csv file in the **data/results.csv** path. You can change this path with the *--path_results* argument ( you can see the list of the full arguments in the *config.py* module). 

### Training
To train the classifier from scratch , use the following command :
```shell
$ python train.py
```
By default , you shall provide the the input parquet files in a *data/* directory that must be created in the same workspace as the training module. The training script will do the preprocessing of data then the fastText and the classifier models training  (see the method section for more details). 
### Evaluation 
To evaluate the model on the test set , use the following command :
```shell
$ python evaluate.py
```
The test set is generated automatically in the training phase. The evaluation module will output two metrics : 
* **The exact accuracy**. for example ,  for a target of [5 , 10 , 19] and a prediction of [5 , 10 , 19] the score will be 1 whereas for a prediction equal to [5 , 10 , 18] the score will be 0. 
* **IoU score**. This score is inspired from the object detection intersection over union score. It measures the overlap between the target and the prediction. It is calculated as the intersection of the two sets over the union. If the target is [5 , 10 , 19] and the label is [5 , 10 , 18] the IoU score will be 2/3. 

# Method 
The pipline of the model includes three main steps :
### Preprocessing 
In this step we mainly extract relevant informations from the URLs. This include the domain name , the path components... We also remove numbers and stopwords and perform stemming. You can refer to *load_data.py* script for more details. 
### Embeddings generation
We use fasttext library to generate word embeddings by the skipgram model. The embedding of an URL is a weighted mean of its word embeddings , where the weights are decreased through the sentence to give more importance to the first words. 
We also benchmark another method that uses contrastive learning to construct URL embeddings. The idea is to train a neural network that minimizes a certain distance between URLs that share at least one label , and increases the distance between URLs that have different labels. Refer to the notebook *contrastive_embeddings.ipynb* for more details. 
### Embeddings generation
Given the embeddings obtained after the previous step , we train a multi-label classifier. We tried two models : Multilabel k Nearest Neighbours ( see  [ref](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/pr07.pdf) ) and Multi-label ARAM (see [ref](https://www.researchgate.net/publication/294088777_HARAM_a_hierarchical_ARAM_neural_network_for_large-scale_text_classification)). The choice of these classifiers is mainly due to the relative low execution time for training and inference of these methods compared to the other multi-label classifiers such as MLSVM. By default , we train the Multi-label ARAM model.
### Rule-based models
We also implemented a rule based model that predicts the labels based on their frequency in the training set. Concretely , if in a given URL there are multiple words that are associated with a certain label (they co-occcur frequently) , This label have higher probability to be predicted. Refer to RuleBased in models.py for more informations. 

# Benchmark 
We benchmark the presented models using the IoU metric. 

| Method        | IoU score           
| ------------- |:-------------:|
| Fasttext + MLARAM     | 0.354 | 
| Fasttext + MLKNN      | 0.347 |  
| Rule-based | 0.335      |
| contrastive learning | 0.267      |

Overall the results are average but this is mainly to the high number of labels in the dataset (1903 categories) and the fact that the URLs by themselves are not very informative. Hence a possible improvement is to merge certain categories that share the same theme and also  extract some header informations from the URLs such as the description to increase its informative capacity. 
