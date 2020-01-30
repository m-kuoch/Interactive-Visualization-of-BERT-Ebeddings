#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:58:18 2019

@author: michael
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltkb

from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper as cm
from bokeh.palettes import Category20
from bokeh.plotting import figure, show, output_file

import TestEmbeddings
import umap

# This function creates an interactive plot of the given "word", created by scanning the dataset at "filename"
def UMAPRepresentation(word, filename, interactive = True):
    
    #Get list of vectors
    vectors = TestEmbeddings.GetEmbeddings(word, 11, filename)
    
    #Convert to array of correct format
    vectorArray = np.array([i for i in vectors[0]])
    
    for i in range (1,len(vectors)):
        vectorArray = np.vstack([vectorArray, [j for j in vectors[i]]])
    
    #print(vectorArray) #To see the vectors for debugging purposes
    
    #Get parts of speech here and match with sentence:
    
    df = pd.read_csv(filename, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values

    posData = []
    for i in sentences:
        if " " + word + " " in i:
            text = nltk.word_tokenize(i)
            pos = nltk.pos_tag(text)
            try:
                wordIdx = text.index(word)
                posTag = pos[wordIdx][1]
            except:
                posTag = 'None'
            posData.append({"sentence": i,"pos": posTag})
    
    #Create UMAP representation here
    sns.set(style="white", context="notebook", rc={"figure.figsize":(14,10)})
    
    reducer = umap.UMAP()
 
    try:

        #Dimension reduction with UMAP
        embedding = reducer.fit_transform(vectorArray)
        #Create the plot
        for i in range(len(embedding)):
            #Get color for part of speech here, implement later if needed:
            plt.scatter(embedding[i][0],embedding[i][1])
            plt.gca().set_aspect('equal', 'datalim')
            plt.title('UMAP projection of the BERT Embeddings for word ' + "\"" + word + "\"" , fontsize=24)

    except:
        print("Error! Possibly not enough sentences with " + "\"" + word + "\"" + " were found in the dataset?")
            
    #Convert to list for easier manipulation in other tasks
    embedding = embedding.tolist()    
    
    data = (posData, embedding)
    
    #Interactive Representation
    if interactive:
        sentences = [data[0][i]["sentence"] for i in range(len(data[0]))]
        pos = [data[0][i]["pos"] for i in range(len(data[0]))]
        coodX = [data[1][i][0] for i in range(len(data[1]))]
        coodY = [data[1][i][1] for i in range(len(data[1]))]
    
        source = ColumnDataSource(dict(x=coodX,y=coodY,label=pos, desc=sentences))
        
        #Create hover data here
        hover = HoverTool(tooltips=[("LOS", "@label"), ("Sentence:", "@desc"),])
        
        #Set different color for each part of speech
        mapper = cm(factors=["CC","CD","IN","EX","FW","DT","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","RB","PRP","PRP","RBR","RBS","RP","TO","UH","VB","VBD","VBG","VBN", "VBP","VBZ","WDT","WP", "WP$", "WRB"],
                    palette=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000', Category20[9][2], Category20[10][0], Category20[10][1], Category20[10][2],
                             Category20[11][0], Category20[11][1], Category20[11][2], Category20[12][0], Category20[12][1], Category20[12][2], Category20[13][0], Category20[13][1], Category20[13][2], Category20[14][0], Category20[14][1]])
    
        p = figure(plot_width=1500, plot_height=800, tools=[hover], title="UMAP")
        p.circle(x='x', y='y', size=10, source=source, color={'field': 'label', 'transform': mapper},legend='label')
    
        output_file('test.html')
        show(p)
        
    return data

#Run the function below to make the plot
#To use a different dataset, simply change "in_domain_train.tsv" to your dataset path

UMAPRepresentation("book", "in_domain_train.tsv")



