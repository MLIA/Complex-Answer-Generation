from trec_car_tools.python3.trec_car.read_data import *
import numpy as np
#import matplotlib.pyplot as plt
#import re
import nltk.data
import pandas as pd
from BM25 import *
import os
import argparse

"""
args: test_folder, train_folder
"""
test_folder="Data/benchmarkY1-test/"
train_folder="Data/benchmarkY1-train/"
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 

def get_para_corpus(file):
    paragraph_corpus={} # (paragraph Id : text)
    with open(file, 'rb') as f:
        for p in iter_paragraphs(f):


            texts = [elem.text if isinstance(elem, ParaText)
                     else elem.anchor_text
                     for elem in p.bodies]
            para_text=' '.join(texts)

            if(p.para_id not in paragraph_corpus.keys()):
                paragraph_corpus[p.para_id] = para_text
            else:
                print("DOUBLONS")
    return paragraph_corpus

def construct_article_first_sent(file,corpus,ranker,reduced_set=[], reduced=False):
    data_heading=[]
    data_no_heading=[]
    with open(file, 'rb') as f:
        for p in iter_pages(f):
            if(not reduced or (reduced and (p.page_id in reduced_set))):
                intro_list=[]
                sent=""
                if(p.child_sections != []):
                    first=p.child_sections[0].get_text_with_headings(False)
                    psg=p.skeleton[0].get_text_with_headings(False)
                      
                    i=1
                    
                    while(psg != first and i < len(p.skeleton) ):
                        sentences=tokenizer.tokenize(psg)
                        if(sentences != []):
                            sent=sentences[0]
                            intro_list.append(sent)
                        psg=p.skeleton[i].get_text_with_headings(False)
                        
                        
                        i+=1
                if(intro_list != []):        
                    intro= "\n".join(intro_list)  
                else:
                    intro=""
                
                query=p.page_name
                query_forBM25=query.replace('/',' ') 
    
                candidats_ids=ranker.rank_corpus({'req':query_forBM25},top_k=10)
                candidats=[corpus[para_id] for para_id in list(candidats_ids['req'].keys())]
                
                answer1=intro+"\n"
                answer2=intro+"\n"
                outline=[]
                if len(p.outline())>0:
                    outline=["/".join([str(section.heading) for section in sectionpath]) for sectionpath in p.flat_headings_list()]
                    ######## Heading
                    for i in range(len(p.outline())):
                        answer1+=p.outline()[i].__str__()
                    ######## no heading
                    headings = p.nested_headings()
                    for (section, _) in headings:
                        psg=section.get_text().split('\n')
                        for t in psg:
                            sent=tokenizer.tokenize(t)
                            if(sent != []):
                                answer2+=sent[0]+' \n'
                        
                data_heading.append({'query':query,'candidats':candidats,'text':answer1,'outline':outline})
                data_no_heading.append({'query':query,'candidats':candidats,'text':answer2,'outline':outline})
    return data_heading,data_no_heading
            
                       
def construct_section_all_passage(file,corpus,ranker,reduced_set=[], reduced=False):
    data_heading=[]
    data_no_heading=[]
    with open(file, 'rb') as f:
        for p in iter_pages(f):
            if(not reduced or (reduced and (p.page_id in reduced_set))):
                if len(p.outline())>0:
    
                    for section in p.child_sections:
                        if section:
                            outline=["/".join([str(s.heading) for s in sectionpath[sectionpath.index(section)+1:len(sectionpath)]]) for sectionpath in p.flat_headings_list() if section in sectionpath]
                            outline.remove('')
                            #### heading
                            answer1=section.str_full_(1)
                            #### no heading
                            answer2=section.get_text()
                            
                            query=p.page_name+'/'+section.heading
                            query_forBM25=query.replace('/',' ')
                            candidats_ids=ranker.rank_corpus({'req':query_forBM25},top_k=10)
                            candidats=[corpus[para_id] for para_id in list(candidats_ids['req'].keys())]
                            data_heading.append({'query':query,'candidats':candidats,'text':answer1,'outline':outline})
                            data_no_heading.append({'query':query,'candidats':candidats,'text':answer2,'outline':outline})
    return data_heading,data_no_heading
def construct_section_first_sent(file,corpus,ranker,reduced_set=[], reduced=False):
    data_heading=[]
    data_no_heading=[]
    with open(file, 'rb') as f:
        for p in iter_pages(f):
            if(not reduced or (reduced and (p.page_id in reduced_set))):
                if len(p.outline())>0:
                    for section in p.child_sections:
                        if section:
                            outline=["/".join([str(s.heading) for s in sectionpath[sectionpath.index(section)+1:len(sectionpath)]]) for sectionpath in p.flat_headings_list() if section in sectionpath]
                            outline.remove('')
                            #### heading
                            answer1=section.str_(1)
                            #### no headding
                            psg=section.get_text().split('\n')
                            answer2=""
                            for t in psg:
                                sent=tokenizer.tokenize(t)
                                if(sent != []):
                                    answer2+=sent[0]+' \n'
                            
                            
                            query=p.page_name+'/'+section.heading
                            query_forBM25=query.replace('/',' ')
                            candidats_ids=ranker.rank_corpus({'req':query_forBM25},top_k=10)
                            candidats=[corpus[para_id] for para_id in list(candidats_ids['req'].keys())]
                            data_heading.append({'query':query,'candidats':candidats,'text':answer1,'outline':outline})
                            data_no_heading.append({'query':query,'candidats':candidats,'text':answer2,'outline':outline})
    return data_heading,data_no_heading
def get_para_corpus_split(folder,n):
    paragraph_corpus={}
    for i in range(1,n+1):
        file=f'{folder}/paragraphCorpus-{i}.cbor'
        paragraph_corpus.update(get_para_corpus(file))
    return paragraph_corpus
    
def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainCorpus', dest='trainCorpus', required=True,
                        help='Train Paragraph corpus')
    parser.add_argument('--testCorpus', dest='testCorpus', required=False,
                        help='Test Paragraph corpus')
    parser.add_argument('--trainPagesFile', dest='trainPagesFile', required=True,
                        help='Train pages folder')
    parser.add_argument('--testPagesFile', dest='testPagesFile', required=False,
                        help='Test pages folder')
    parser.add_argument('--reducedset', dest='reducedset', required=False,
                        help='Train set reduction')
    parser.add_argument('--fold', dest='fold', required=True,
                        help='Number of the train fold')
    parser.add_argument('--output-folder', dest='output_folder', default='.',
                        help='Output folder')

    args = parser.parse_args()

    output_folder=args.output_folder+'/output_datasets'
    if not os.path.exists(output_folder): 
        print('Creating folder to store preprocessed dataset at:')
        print(output_folder)
        os.mkdir(output_folder)
    if not os.path.exists(f'{output_folder}/fold-{args.fold}'): 
        print('Creating folder to store processed dataset at:')
        print(f'{output_folder}/fold-{args.fold}')
        os.mkdir(f'{output_folder}/fold-{args.fold}')
    if not os.path.exists(f'{output_folder}/RI_index-{args.fold}'): 
        print('Creating folder to store indexes:')
        print(f'{output_folder}/RI_index-{args.fold}')
        os.mkdir(f'{output_folder}/RI_index-{args.fold}')
        
 
    for setname in ['train']:

        if(setname == 'train' ):
            corpus_filename=args.trainCorpus
            pages_filename=args.trainPagesFile
        else:
            corpus_filename=args.testCorpus
            pages_filename=args.testPagesFile
        index_filename=f'{output_folder}/RI_index-{args.fold}/{setname}_passages_corpus_index'
        output1=f'{output_folder}/fold-{args.fold}/article_1st_sent_no_heading_{setname}.csv'
        output2=f'{output_folder}/fold-{args.fold}/sections_all_psg_no_heading_{setname}.csv'
        output3=f'{output_folder}/fold-{args.fold}/sections_1st_sent_no_heading_{setname}.csv'
        
        
        output12=f'{output_folder}/fold-{args.fold}/article_1st_sent_heading_{setname}.csv'
        output22=f'{output_folder}/fold-{args.fold}/sections_all_psg_heading_{setname}.csv'
        output32=f'{output_folder}/fold-{args.fold}/sections_1st_sent_heading_{setname}.csv'
        paragraph_corpus=get_para_corpus(corpus_filename)

        if not os.path.exists(index_filename): 
            BM25.create_index(paragraph_corpus, index_filename)
        
        ranker = BM25(index_filename)
    
            
        d_H,d_nH=construct_article_first_sent(pages_filename,paragraph_corpus,ranker)
        data_heading=pd.DataFrame(d_H)
        data_no_heading=pd.DataFrame(d_nH)
        data_no_heading.to_csv(output1)
        data_heading.to_csv(output12)

        d_H,d_nH=construct_section_all_passage(pages_filename,paragraph_corpus,ranker)
        data_heading=pd.DataFrame(d_H)
        data_no_heading=pd.DataFrame(d_nH)
        data_no_heading.to_csv(output2)
        data_heading.to_csv(output22)

        d_H,d_nH=construct_section_first_sent(pages_filename,paragraph_corpus,ranker)
        data_heading=pd.DataFrame(d_H)
        data_no_heading=pd.DataFrame(d_nH)
        data_no_heading.to_csv(output3)
        data_heading.to_csv(output32)

if __name__ == '__main__':
    main()
