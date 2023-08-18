# Complex Answer Generation For Conversational Search Systems. 

Code for [Does Structure Matter? Leveraging Data-to-Text Generation for Answering Complex Information Needs]()

Transformers implementation is based on [Huggingface](https://huggingface.co/) 

## Datasets

TREC CAR dataset is used: benchmarkY1test for testing and Large-scale training data for training. Download from : http://trec-car.cs.unh.edu/datareleases/index.html#:~:text=July%2026%2C%202018).-,benchmarkY1train,-benchmarkY1%2Dtrain.v2.0

Various adaptations are created using data_construction/make_data.py file. The data construction uses [TREC CAR TOOLS](https://github.com/TREMA-UNH/trec-car-tools) (must be cloned/copied in data-contruction folder) 

## Models

cogecsea folder contains implementation of: a finetuned T5 (from huggingface), a sequential planning-model based on T5, and an end-to-end planning model using T5. 
