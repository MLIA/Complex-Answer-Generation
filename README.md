# Complex Answer Generation For Conversational Search Systems. 

Code for [Does Structure Matter? Leveraging Data-to-Text Generation for Answering Complex Information Needs]()

Most of this code is based on [Huggingface](https://huggingface.co/) 

## Datasets

TREC CAR dataset is used: benchmarkY1test for testing and Large-scale training data for training. Download from : [http://trec-car.cs.unh.edu/ ](http://trec-car.cs.unh.edu/datareleases/index.html)

Various adaptations are created using data_construction/make_data.py file. The data construction uses [TREC CAR TOOLS](https://github.com/TREMA-UNH/trec-car-tools) (must be cloned/copied in data-contruction folder) 

## Models

cogecsea folder contains implementation of: a finetuned T5 (from huggingface), a sequential planning-model based on T5, and an end-to-end planning model using T5. 
