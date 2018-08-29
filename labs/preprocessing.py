#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:31:54 2018
@author: marlene

PREPROCESSING MODULE FOR THE "TRIANGLES" DATASET
"""

###########
# IMPORTS #
###########
import numpy as np
import pandas as pd
import glob, re


##########
# MODULE #
##########
class Preprocessing:
    """
    Module for pre-processing text data (triangles data set) into count data
    (raw counts for bag-of-words features; word collocations for distributed
    semantics representations).
    """

    def __init__(self):
        self.count_data = {} # a dict to store all the data (words)
        self.collocations = {} # a dict to store the collocations
        self.num_words = 0 # keep track of the total number of words


    def reset(self):
        """
        Reset the internally stored data and word count.
        """
        self.count_data = {}
        self.collocations = {}
        self.num_words = 0

    def remove_single_words(self, count_dict):
        for word in count_dict.keys():
            if count_dict[word] < 2:
                del count_dict[word]
        return count_dict


    def get_wordcounts(self,subject,transcript):
        """
        Get simple wordcounts for bag-of-words models.
        """
        #first add a dictionary per id/subject. this will store the word counts
        if subject not in self.count_data:
            self.count_data[subject] = {}
        # now iterate over words and keep only the counts
        for word in transcript.split():
            self.num_words +=1
            #do some cleaning up
            word = word.lower().strip()
            #let's be strict and only keep WORDS made of LETTERS
            word = re.sub('[^a-zA-Z]', '', word)
            if word not in self.count_data[subject]:
                self.count_data[subject][word] = 0
            self.count_data[subject][word] += 1


    def get_collocations(self,subject,transcript):
        """
        Get counts per 'document', for distributed semantics models.
        """
        pass


    def preprocesses_schizophrenia_data(self,filepath):
        """
        Parse out participant, group and text information in the way that matches
        the organisation of the schizophrenia data files.
        lines are organised as follows (tab seperated file):
        Subject	Task	StartTime	Transcript	EndTime	Conservative
        """
        data = open(filepath).readlines()
        for line in data[1:]: #skip the header line
            lnsplit = line.split("\t")
            #task = lnsplit[1]
            transcript = lnsplit[3]
            group = filepath[43:-4] #parse out the group name
            subject = group+lnsplit[0]
            self.get_wordcounts(subject,transcript)
            self.get_collocations(subject,transcript)


    def preprocesses_depression_data(self,filepath):
        """
        Parse out participant, group and text information in the way that matches
        the organisation of the depression data files.
        lines are organised as follows (tab seperated file):
        'TimeStart', 'Sentence', 'TimeStop', 'File'
        (where file is the participant/group ID (?))
        """
        data = open(filepath).readlines()
        # lines are organised: Subject	Task	StartTime	Transcript	EndTime	Conservative
        for line in data[1:]: #skip the header line
            lnsplit = line.split("\t")
            transcript = lnsplit[1]
            subject = lnsplit[3].strip()[:-4]
            self.get_wordcounts(subject,transcript)
            self.get_collocations(subject,transcript)


    def write_df(self, writepath, depression=False):
        """
        Create a dataframe from nested dictionaries and write it to a csv file.
        :param writepath: name of path to save dataframe (csv)
        :param depression: whether or not the data is depression data (default: schizophrenia)
        """
        vocab = sorted(list(self.get_full_vocab()))
        print(len(vocab))
        print(self.num_words)
        #turn the data into a dataframe
        df = pd.DataFrame(columns=vocab)
        df = pd.DataFrame.from_dict(self.count_data, orient="index")
        df = df.fillna(0) #fill the NAs with 0s -- unobserved words are 0 counts
        #add labels (instead of IDs)
        if depression:
            df['label'] = [re.sub("[0-9]", "", name) for name in df.index]
        else:
            df['label'] = [name[13:21] for name in df.index]
        df = df.reset_index()
        df = df.drop('index', axis=1)
        #drop words that only appear once in the whole dataset
        for col in df.columns:
            if len(df[df[col]!=0]) <2:
                df.drop(col,inplace=True,axis=1)
        print(df.shape)
        #print(df.head())
        #write to csv
        df.to_csv(writepath)


    def get_full_vocab(self):
        """
        Get the set of unique words across the dataset. This is useful for turning
        (dense) word count dictionaries into (sparse) feature vectors.
        """
        vocab = set()
        for _, subject in self.count_data.items():
            vocab.update(subject.keys())
        return vocab


########
# MAIN #
########
def main():
    path_to_data = "/home/marlene/Dropbox/TrianglesTranscripts"
    proc_s = Preprocessing()
    proc_d = Preprocessing() #have a seperate object for depressed and schizophrenic

    for filename in glob.glob(path_to_data+"/*"):
        #the schizophreania and depression data are organised in a different way
        if filename == path_to_data+"/DepressionTriangles.txt":
            proc_d.preprocesses_depression_data(filename)
        else:
            proc_s.preprocesses_schizophrenia_data(filename)

    #write the raw counts to dataframes
    proc_d.write_df("../data/triangles_depression.csv",depression=True)
    proc_s.write_df("../data/triangles_schizophrenia.csv")

    #write the collocations to dataframes


#run
if __name__ == "__main__":
    main()
