'''
This file contains the methed GetEmbeddings, which takes in a dataset of sentences
and returns the latent vector representations of the words at the given layer of the model
'''

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np
import pandas as pd

def GetEmbeddings(word, layer, filename): # Word is word of choice and layer is an int in range 0-11

    # If you would like to use a different model than the pretrained one here, simply replace
    # the line below with the model you wish to use
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    '''
    #Open file
    f = open(filename, 'r')
    #Create a list of lines
    lines = f.readlines()
    
    #List to store word vectors
    embeddings = []

    #Retrive pretrained model and turn of training
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    i = 0
    numSamples = 0
    
    '''
    
    #Retrive pretrained model and turn of training
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    #List to store word vectors
    embeddings = []
    
    #.tsv files version:
    df = pd.read_csv(filename, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    sentences = df.sentence.values
    
    #Count the number of sentences with word:
    count = 0
    
    #For each sentence i in the dataset, get the word vector and append to list of vectors
    #*Only if the word is in the sentence
    for i in sentences:
        if " " + word + " " in i:
            
            #Add special tokens to the input text and tokenize it:
            markedText = "[CLS] " + i + " [SEP]"
            tokenizedText = tokenizer.tokenize(markedText)

            #Which index is the word at?
            wordIndex = tokenizedText.index(word)
            
            #Match the tokens to the id's in vocab list
            tokenIds = tokenizer.convert_tokens_to_ids(tokenizedText)
            
            #Specify single or two sentence input:
            segmentIds = [1] * len(tokenizedText)
            #print(segmentIds)
            
            #Convert to pytorch tensor for model
            tokensTensor = torch.tensor([tokenIds])
            segmentsTensors = torch.tensor([segmentIds])
            
            
            with torch.no_grad():
                encodedLayers = model(tokensTensor, segmentsTensors)[0]
    
            #Get vector at given layer
                
            # Lookup the vector for token at wordIndex (the word) in layer
            vector = encodedLayers[layer][0][wordIndex]
    
            embeddings.append(vector)
            
            count += 1
            
        #If 100 sentences with the word have already been found, then break the loop
        if count >= 200:
            break
    
    #Print the number of sentences found
    print("Number of sentences with " + "\"" + word + "\"" + ": " + str(count))
        
    embeddingsList = []
    for i in range(len(embeddings)):
        embeddingsList.append([j.item() for j in embeddings[i].flatten()])
    return embeddingsList

#Test here:
#print(GetEmbeddings("bank", 11, "bertTestData.txt"))
        
