import os
import torch

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import random

NUM_PSG=10

class DTDataset(object):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        _, query, candidats, label, outline = self.dataframe.iloc[index] 
        #query, candidats, label = self.dataframe.iloc[index] 
        return query, candidats, label,outline

    def get_csv_row(self, index):
        _, query, candidats, label, outline = self.dataframe.iloc[index] 
        #query, candidats, label = self.dataframe.iloc[index]
        return query, candidats, label, outline

class DTDatasetFormat(DTDataset):
    def __getitem__(self, index):
        query, candidats, label = self.get_csv_row(index) #query, candidats, label, outline = #self.get_csv_row(index)
        # print ensuring all str
        input_text = 'question: '+ query + ' [Documents:] '+ ''.join(['[Document:]' + candidat for candidat in candidats]) 
        new_query = 'question: ' + query
        """
        input_text = '[Query:] '+ query + ' [Documents:] '+ ''.join(['[Document:]' + candidat for candidat in candidats]) 
        new_query = '[Query:] ' + query
        
        input_text = query + ' [Documents:] '+ ''.join(['[Document:]' + candidat for candidat in candidats])
        new_query =  query
        """
        return new_query, input_text, label


    @classmethod
    def construct_from(cls, data_folder, data_seed=42, data_split=0.01,
                       nrows_train_val=None, nrows_test=None, only_outline=False):
        ''' Construct the default dataset returnin the train/val/test datasets.

            Parameters
            ----------
            data_folder : str
                Path to dataset folder
            data_seed : int
                The seed used to split train and validation set
            data_split : float, optional
                The ratio of the validation set according to train set size
            nrows_train_val : int, optional
                The maximum number of rows to load from the csv for training and
                 validation set
            nrows_test : int, optional
                The maximum number of rows to load from the csv for testing set

            only_outline : boolean, optional
                The labels or output sequence is only the outlines if True
        '''
        def detect_outline(x):
            out = x.split('/')
            return '[h'+str(len(out))+']'+out[-1]+ '[/h'+str(len(out))+']'

        training_set = pd.read_csv(os.path.join(data_folder, 'train.csv'), encoding='utf-8',
                                   converters={"candidats": eval, 'outline':eval}, nrows=nrows_train_val)
        training_set.dropna(inplace=True)
        training_set = training_set[training_set['outline'].map(len)>0]
        training_set = training_set[training_set['text']!= '\n']
        training_set = training_set[training_set['candidats'].map(len)>0]
        if only_outline:
            training_set = training_set[training_set['outline'].map(len)>0]
            training_set['text'] = training_set['outline'].apply(lambda x : ''.join(map(detect_outline, x)))

        torch.manual_seed(data_seed)
        random_permutation = torch.randperm(len(training_set))
        val_set_index = round(data_split * len(training_set))


        # building training set
        rp_train = random_permutation[val_set_index:].tolist()
        sub_training_set = training_set.iloc[rp_train]
        train_dt = cls(sub_training_set)

        # building validation set
        rp_val = random_permutation[:val_set_index].tolist()
        valing_set = training_set.iloc[rp_val]
        val_dt = cls(valing_set)

        # building test_set
        testing_set = pd.read_csv(os.path.join(data_folder, 'test.csv'), encoding='utf-8',
                                  converters={"candidats": eval, 'outline':eval}, nrows=nrows_test)
        testing_set.dropna(inplace=True)
        testing_set = testing_set[testing_set['candidats'].map(len)>0]
        testing_set['text']= testing_set['text']
        if only_outline:
            testing_set = testing_set[testing_set['outline'].map(len)>0]
            testing_set['text'] = testing_set['outline'].apply(lambda x : ''.join(map(detect_outline, x)))
        test_dt = cls(testing_set)

        return train_dt, val_dt, test_dt
    
    @classmethod
    def construct_mlm_from(cls, nrows_train_val,save_folder=""):

        dataset_all = load_dataset("wikicorpus", "raw_en")
        dataset_all=dataset_all['train'].shuffle()
        if(nrows_train_val is not None):
            dataset=dataset_all[0:nrows_train_val]
            dataset=Dataset.from_dict(dataset)
            dataset_nsent=dataset_all[nrows_train_val:nrows_train_val+nrows_train_val]
            dataset_nsent=Dataset.from_dict(dataset_nsent)
        dataset=dataset.map(lambda x: {'id':x['id'],'title':x['title'],'text': x['text'].replace("\n"," ")})
        dataset_nsent=dataset_nsent.map(lambda x: {'id':x['id'],'title':x['title'],'text': x['text'].replace("\n"," ")})
        candidats=[]
        text=[]
        query=list(map(lambda x: '[MLM] [Query:]'+x, dataset['title'])) +list(map(lambda x: '[NEXT] [Query:]'+x, dataset_nsent['title']))
        for r in range(dataset.num_rows):
            words=dataset[r]['text'].split()
            ln=len(words)//NUM_PSG
            psg=[" ".join(words[i*ln:(i+1)*ln]) for i in range(NUM_PSG) ]
            index=random.randint(0, NUM_PSG-1)
            text.append(psg[index])
            psg[index]="<UNK>"
            candidats.append(psg)
        for r in range(dataset_nsent.num_rows):
            words=dataset_nsent[r]['text'].split()
            ln=len(words)//NUM_PSG
            index=random.randint(0, NUM_PSG-2)
            candidats.append([" ".join(words[index*ln:(index+1)*ln])])
            text.append(" ".join(words[(index+1)*ln:(index+2)*ln]))
        d = {'query': query, 'candidats': candidats,'text':text}
        df_train = pd.DataFrame(data=d)  
        df_train.to_csv(save_folder+"pretrain_dataset.csv")
        return cls(df_train), cls(df_train[0:6]), cls(df_train[0:6])


class DTDatasetFormatTVT(DTDataset):
    def __getitem__(self, index):
        query, candidats, label, outline = self.get_csv_row(index)
        # print ensuring all str
        input_text = 'question: '+ query + ' [Documents:] '+ ''.join(['[Document:]' + candidat for candidat in candidats])  
        new_query = 'question: ' +query 
        """
        input_text = '[Query:] '+ query + ' [Documents:] '+ ''.join(['[Document:]' + candidat for candidat in candidats])  
        new_query = '[Query:] ' +query 
        input_text =  query + ' [Documents:] '+ ''.join(['[Document:]' + candidat for candidat in candidats])  
        new_query = query 
        """
        return new_query, input_text, label


    @classmethod
    def construct_from(cls, data_folder, data_seed=42, data_val_split=0.01, data_test_split=0.01,
                       nrows_train_val=None, only_outline=False):
        ''' Construct the default dataset returnin the train/val/test datasets.

            Parameters
            ----------
            data_folder : str
                Path to dataset folder
            data_seed : int
                The seed used to split train and validation set
            data_val_split : float, optional
                The ratio of the validation set according to train set size
            data_val_split : float, optional
                The ratio of the set set according to train set size
            nrows_train_val : int, optional
                The maximum number of rows to load from the csv 
            only_outline : boolean, optional
                The labels or output sequence is only the outlines if True
        '''
        def detect_outline(x):
            out = x.split('/')
            return '[h'+str(len(out))+']'+out[-1]+ '[/h'+str(len(out))+']'

        training_set = pd.read_csv(os.path.join(data_folder, 'train.csv'), encoding='utf-8',
                                   converters={"candidats": eval, 'outline':eval}, nrows=nrows_train_val)
        training_set.dropna(inplace=True)
        training_set = training_set[training_set['outline'].map(len)>0]
        training_set = training_set[training_set['text']!= '\n']
        training_set = training_set[training_set['candidats'].map(len)>0]
        if only_outline:
            training_set = training_set[training_set['outline'].map(len)>0]
            training_set['text'] = training_set['outline'].apply(lambda x : ''.join(map(detect_outline, x)))

        
        # getting query to ensure same articles are not in train and test
        queries = [query.split('/')[0] for query in training_set['query']]
        training_set['root-queries'] = queries


        # uniqueness of queries 
        queries_set = set()
        queries_list = []
        # use this function do avoid random order with set
        for q in queries:
            if q not in queries_set:
                queries_list.append(q)
                queries_set.add(q)

        queries = np.array(queries_list)

        torch.manual_seed(data_seed)
        random_permutation = torch.randperm(len(queries))
        randomized_queries = queries[random_permutation]

        val_index_split = round(len(queries) * data_val_split)
        test_index_split = val_index_split + round(len(queries) * data_test_split)

        val_queries = randomized_queries[:val_index_split]
        test_queries = randomized_queries[val_index_split:test_index_split]
        train_queries = randomized_queries[test_index_split:]

        training_set = training_set.set_index('root-queries')

        # building validation set
        valing_set = training_set.loc[val_queries]
        val_dt = cls(valing_set)

        # building test set
        sub_testing_set = training_set.loc[test_queries]
        test_dt = cls(sub_testing_set)

        # building training set
        sub_training_set = training_set.loc[train_queries]
        train_dt = cls(sub_training_set)

        return train_dt, val_dt, test_dt
    
    @classmethod
    def construct_mlm_from(cls, nrows_train_val,save_folder=""):

        dataset_all = load_dataset("wikicorpus", "raw_en")
        dataset_all=dataset_all['train'].shuffle()
        if(nrows_train_val is not None):
            dataset=dataset_all[0:nrows_train_val]
            dataset=Dataset.from_dict(dataset)
            dataset_nsent=dataset_all[nrows_train_val:nrows_train_val+nrows_train_val]
            dataset_nsent=Dataset.from_dict(dataset_nsent)
        dataset=dataset.map(lambda x: {'id':x['id'],'title':x['title'],'text': x['text'].replace("\n"," ")})
        dataset_nsent=dataset_nsent.map(lambda x: {'id':x['id'],'title':x['title'],'text': x['text'].replace("\n"," ")})
        candidats=[]
        text=[]
        query=list(map(lambda x: '[MLM] [Query:]'+x, dataset['title'])) +list(map(lambda x: '[NEXT] [Query:]'+x, dataset_nsent['title']))
        for r in range(dataset.num_rows):
            words=dataset[r]['text'].split()
            ln=len(words)//NUM_PSG
            psg=[" ".join(words[i*ln:(i+1)*ln]) for i in range(NUM_PSG) ]
            index=random.randint(0, NUM_PSG-1)
            text.append(psg[index])
            psg[index]="<UNK>"
            candidats.append(psg)
        for r in range(dataset_nsent.num_rows):
            words=dataset_nsent[r]['text'].split()
            ln=len(words)//NUM_PSG
            index=random.randint(0, NUM_PSG-2)
            candidats.append([" ".join(words[index*ln:(index+1)*ln])])
            text.append(" ".join(words[(index+1)*ln:(index+2)*ln]))
        d = {'query': query, 'candidats': candidats,'text':text}
        df = pd.DataFrame(data=d)    
        df.to_csv(save_folder+"pretrain_dataset.csv")
        return df, None, None


class DTDatasetFormatContextList(DTDatasetFormat):
    def __getitem__(self, index):
        query, candidats, label, outline = self.get_csv_row(index)
        input_text = [f'question: '+ query + ' [Document:]' + candidat for candidat in candidats]
        new_query = 'question: ' +query
        """
        input_text = [f'[Query:] '+ query + ' [Document:]' + candidat for candidat in candidats]
        new_query = '[Query:] ' +query 
        
        input_text = [query + ' [Document:]' + candidat for candidat in candidats] 
        new_query = query 
        """
        return new_query, input_text, label

class DTDatasetFormatContextListTVT(DTDatasetFormatTVT):
    def __getitem__(self, index):
        query, candidats, label, outline = self.get_csv_row(index)
        input_text = [f'question: '+ query + ' [Document:]' + candidat for candidat in candidats] 
        new_query = 'question: ' +query 
        """
        input_text = [f'[Query:] '+ query + ' [Document:]' + candidat for candidat in candidats] 
        new_query = '[Query:] ' +query 
        
        input_text = [query + ' [Document:]' + candidat for candidat in candidats] #f'[Query:] '+ 
        new_query = query # '[Query:] ' +
        """
        return new_query, input_text, label


class DTDatasetFormatNoTraining(DTDatasetFormat):
    ''' A variant of DTDatasetFormat to perform better on non-pretrained model.

        If you use a summarization based model without training we recommand to use
        this dataset. This dataset do not add the specific tokens on the candidats.
        Those ones can potentially lower the performances being interpreted as important.
    '''
    def __getitem__(self, index):
        query, candidats, label, outline = DTDataset.__getitem__(self, index)
        # print ensuring all str

        input_text =  query + '. ' + ''.join([str(candidat) for candidat in candidats])
        new_query = query+'. '
        return new_query, input_text, label

class OutlineTextDataset(DTDataset):
    @classmethod
    def construct_from(cls, train_val_test_queries,
                       data_path='/net/big/gerald/data/data2text/queries_paragraph_outlines.csv'):
        train, val, test = train_val_test_queries
        train, val, test = train.dataframe, val.dataframe, test.dataframe
        train_queries, val_queries, test_queries = set(train.index), set(val.index), set(test.index)
        df = pd.read_csv(data_path, converters={'outlines':eval})
        df1 = df.set_index('query')

        def get_df(df, queries, queries_df):
            b_queries = list(queries.intersection(set(df.index)))
            train_df = df.loc[b_queries]
            train_df2 = queries_df[~queries_df.index.duplicated(keep='first')]
            train_candidats = list(train_df2.loc[train_df.index]['candidats'])
            train_df['candidats'] = train_candidats
            def make_outline( outline):
                #str_ret = '[TOPIC]'+title+'[\TOPIC]'
                str_ret = ""
                for i, coutline in enumerate(outline):
                    str_ret+='[h'+str(i+1)+']'+coutline+'[/h'+str(i+1)+']'
                return str_ret
            train_df['processed_outlines'] = train_df['outlines'].apply(make_outline)
            return train_df

        train_dt = cls(get_df(df1, train_queries, train))
        val_dt = cls(get_df(df1, val_queries, val))
        test_dt = cls(get_df(df1, test_queries, test))

        return train_dt, val_dt, test_dt

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return row.query_id, row.name, row.candidats, row.relevant_document, row.processed_outlines
