import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd

from collections import namedtuple, deque
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import random
import math

EMBEDDING_DIM = 8

class Tools(object):
     
    def __init__(self):
        super(Tools, self).__init__()
         
    # transfer feature set to embedding
    def feature_state_generation_des(feature_set):
        df = pd.DataFrame(feature_set)
        feature_matrix = []
        for i in range(8):
            feature_matrix = feature_matrix + list(df.astype(np.float64).
                                                describe().iloc[i, :].describe().fillna(0).values)
        return feature_matrix
    
    # features and label are np.arrays
    # row: number samples
    # column: number features
    def cal_mutual_info(features, label):
        mutual_info_list = mutual_info_regression(features, label).tolist() 
        total_mutual = 0.0
        for mutual in mutual_info_list:
            total_mutual += mutual
        return total_mutual / len(mutual_info_list)
    
    def cal_redundancy(feat1, feat2):
        totle_redundancy = 0.0
        for col in range(feat2.shape[1]):
            mutual_info_list = mutual_info_regression(feat1, feat2[:,col]).tolist()
            for mutual in mutual_info_list:
                totle_redundancy += mutual
        return totle_redundancy / feat1.shape[1] / feat2.shape[1]
            
    # all data set: padas dataframe
    # X: number_samples * number_features
    # Y: number_samples * 1
    def machine_learning_task(X_train, X_val, label_train, label_val, mode):
        if mode == "random_forest":
            model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
            model.fit(X_train, label_train)
            accuracy = model.score(X_val, label_val)
            return accuracy
        if mode == "decision_tree_classify":
            model = classifier_tree = DecisionTreeClassifier(max_depth=10)
            model.fit(X_train, label_train)
            accuracy = model.score(X_val, label_val)
            return accuracy
        if mode == "logistic_regression":
            model = LogisticRegression()
            model.fit(X_train, label_train)
            accuracy = model.score(X_val, label_val)
            return accuracy
        
    # get the features' index of the data set
    def action_to_feature_index(train_data):
        title = train_data.columns.tolist()
        action_to_index = {}
        action = 0
        start = 0
        end = 0
        title_dict = {}
        for index in range(len(title)):
            prefix_title = "_".join(title[index].split("_")[:-1])
            if prefix_title not in title_dict:
                title_dict[prefix_title] = 1
                if end != 0:
                    action_to_index[action] = [start, index-1]
                    start = index
                    end = index
                    action += 1
            else:
                end = index
        action_to_index[action] = [start, index]
        return action_to_index
    
    # convert cross feature to embedding
    # action1: meta controller action
    # action2: controller action
    # action: cross feature action
    def origin_feature_to_embedding(origin_feature, action1, action2, action):
        origin_feature = origin_feature.tolist()
        res_dic = {}
        num = 0
        for f in origin_feature:
            if f not in res_dic:
                res_dic[f] = num
                num += 1
        num_dic = len(res_dic)
        input = []
        for feat in origin_feature:
            input.append(hash(feat)%num_dic)
        input = torch.tensor(input, dtype=torch.long)
        embedding = torch.nn.Embedding(num_dic, EMBEDDING_DIM)
        emb_df = pd.DataFrame(embedding(input).tolist())
        # set columns' name
        column = []
        for index in range(emb_df.shape[1]):
            name = "feature_" + str(action1) + "x" + str(action2) +"_"+str(action*EMBEDDING_DIM+index)
            column.append(name)
        emb_df.columns = column
        return emb_df
