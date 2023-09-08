import os
import warnings
from model import DQN
from tools import Tools
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='output.log.random_forest',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 8
MAX_DATA_NUM = 40
NUM_ORIGIN = 20

def data_init(embedding_url, origin_url):
    # load data set
    data_embedding = pd.read_hdf(embedding_url)
    data_origin = pd.read_hdf(origin_url)
    label = data_embedding["ACTION"]
    data_embedding = data_embedding.drop("ACTION", axis=1)
    data_origin = data_origin.drop("ACTION", axis=1)
    
    # create a dict
    # key: action
    # value: index of the feature
    action_to_feature_index = Tools.action_to_feature_index(data_embedding)
    
    # create embedding for every single feature
    # todo: use a new auto encoder to generate embedding
    action_to_embedding = {}
    for key, value in action_to_feature_index.items():
        if key not in action_to_embedding:
            feature = data_embedding.iloc[:,value[0]:value[1]+1]
            action_to_embedding[key] = torch.tensor(
                [Tools.feature_state_generation_des(feature)], device=device, dtype=torch.float32)
    return data_embedding, data_origin, label, action_to_feature_index, action_to_embedding

def feature_cross(data_embedding, data_origin, action_to_index, 
                  action_to_embedding, meta_action, controller_actions, 
                  feature_label_relevance, label):
    # select feature from origin data to do feature cross
    # just concat with the "_"
    meta_feature_ori = data_origin.iloc[:, meta_action]
    for action in controller_actions:
        controller_feature_ori = data_origin.iloc[:, action]
        controller_feature_name = controller_feature_ori.name
        # cross the feature
        cross_feature_ori = meta_feature_ori + "_" + controller_feature_ori
        cross_feature_ori.name = meta_feature_ori.name + "x" + controller_feature_ori.name
        # cross feature embedding
        cross_feature_emb = Tools.origin_feature_to_embedding(cross_feature_ori, 
                                                            meta_action, action, 
                                                            data_origin.shape[1])
        # update dict
        rele = Tools.cal_mutual_info(cross_feature_emb, np.array(label))
        cross_action = len(action_to_index)
        if len(action_to_embedding) < MAX_DATA_NUM:
            feature_label_relevance[cross_action] = rele
        else:
            tmp_dic = sorted(feature_label_relevance.items(), key = lambda x: x[1])
            # just compare the first one
            for (key, value) in tmp_dic:
                # must select origin feature
                if rele > value: #and key >= NUM_ORIGIN:
                    del feature_label_relevance[key]
                    # del action_to_embedding[key]
                    feature_label_relevance[cross_action] = rele
                break    
        action_to_embedding[cross_action] = \
                torch.tensor([Tools.feature_state_generation_des(cross_feature_emb)], 
                            device=device, dtype=torch.float32)    
        action_to_index[cross_action] = [EMBEDDING_DIM*cross_action, EMBEDDING_DIM*(cross_action+1)-1]
        # concat
        data_embedding = pd.concat([data_embedding, cross_feature_emb], axis=1, join="outer")
        data_origin = pd.concat([data_origin, cross_feature_ori], axis=1, join="outer")
    return data_embedding, data_origin, action_to_embedding, action_to_index, feature_label_relevance

def exclude_meta_action(controller_actions, meta_action, topk):
    num_action = 0
    final_controller_actions = []
    for index in range(topk+1):
        if num_action == topk:
            break
        if controller_actions[index] == meta_action:
            continue
        else:
            final_controller_actions.append(controller_actions[index])
            num_action += 1
    return final_controller_actions

def cal_relevance_or_redundancy(meta_feature, data_embedding, label,
                                action_to_feature_index, final_controller_actions, flag):
    np_label = np.array(label)
    np_meta_feature = np.array(meta_feature)
    relevance = 0.0
    redundancy = 0.0
    if flag == "meta_controller":
        relevance = Tools.cal_mutual_info(np_meta_feature, np_label)
    else:
        for action in final_controller_actions:
            feature_index = action_to_feature_index[action]
            selected_feature = np.array(data_embedding.iloc[:,feature_index[0]:feature_index[1]+1])
            # compute relevance
            relevance += Tools.cal_mutual_info(selected_feature, np_label)
            # compute redundancy
            redundancy += Tools.cal_redundancy(np_meta_feature, selected_feature)
    return relevance,redundancy

if __name__ == "__main__":
    
    try:
        # load data
        embedding_url = "data_process/bank/data_embedding.hdf"
        origin_url = "data_process/bank/data_origin.hdf"
        data_embedding, data_origin, label, action_to_feature_index, \
            action_to_embedding = data_init(embedding_url, origin_url)
            
        # init the relevance of the selected feature
        feature_label_relevance_f = {}
        for key, value in action_to_embedding.items():
            feature_index = action_to_feature_index[key]
            selected_feature = np.array(data_embedding.iloc[:,feature_index[0]:feature_index[1]+1])
            rele = Tools.cal_mutual_info(selected_feature, np.array(label))
            feature_label_relevance_f[key] = rele
            
         # parameters for DQN
        meta_controller_state_dim = 64
        meta_controller_action_dim = action_to_embedding[0].shape[1]
        meta_controller_actions = len(action_to_embedding)
        meta_controller_hidden_dim = 100
        meta_controller_init_weight = 0.001
        meta_controller_capacity = 40
        
        controller_state_dim = 64 + action_to_embedding[0].shape[1]
        controller_action_dim = action_to_embedding[0].shape[1]
        controller_actions = len(action_to_embedding)
        controller_hidden_dim = 100
        controller_init_weight = 0.001
        controller_capacity = 40
        
        # init DQN
        meta_controller = DQN(meta_controller_state_dim, meta_controller_action_dim,
                      meta_controller_actions, meta_controller_hidden_dim, 
                      meta_controller_init_weight, meta_controller_capacity, device)
        controller = DQN(controller_state_dim, controller_action_dim, controller_actions,
                 controller_hidden_dim, controller_init_weight, controller_capacity, device)
            
        # train
        episode = 5
        all_iterations = 60
        data_embedding_f = data_embedding.copy()
        for epi in range(episode):
            # use describe to generate the embedding of the feature set
            # todo: use auto encoder to generate the embedding 
            feature_rep = Tools.feature_state_generation_des(data_embedding_f)
            feature_label_relevance = feature_label_relevance_f.copy()
            ACC = 0.0
            FIRST_ITER = 0
            state = torch.tensor([feature_rep], device=device, dtype=torch.float32)
            for iter in range(all_iterations):
                # select action based on meta controller
                meta_action_tensor = meta_controller.select_action(state, action_to_embedding, 1)
                meta_action = meta_action_tensor.tolist()[0][0]

                # describe embedding
                meta_feature_des = action_to_embedding[meta_action]
                
                # concat feature set with a selected feature
                next_controller_state = torch.cat((state, meta_feature_des), 1)
                # push controller
                if iter == FIRST_ITER:
                    controller_state = next_controller_state
                else:
                    controller.push(controller_state, 
                                    torch.tensor([final_controller_actions], device=device, dtype=torch.long), 
                                    next_controller_state, controller_reward)
                    controller_state = next_controller_state
                
                # select actions based on controller
                topk = 1
                controller_actions = controller.select_action(controller_state, action_to_embedding, topk+1)
                
                # exclude meta action from controller actions
                final_controller_actions = exclude_meta_action(controller_actions.tolist()[0], meta_action, topk)
                
                # feature cross
                new_embedding, new_origin, new_aciton_to_emb, new_action_to_index, new_feature_label_relevance = \
                                                        feature_cross(data_embedding, data_origin, 
                                                                    action_to_feature_index, action_to_embedding,
                                                                    meta_action, final_controller_actions,
                                                                    feature_label_relevance, label)
                data_embedding, data_origin, action_to_embedding, action_to_feature_index, feature_label_relevance = \
                                            new_embedding, new_origin, new_aciton_to_emb, new_action_to_index, new_feature_label_relevance
                
                # compute relevance and redundancy
                meta_feature_index = action_to_feature_index[meta_action]
                meta_feature = data_embedding.iloc[:, meta_feature_index[0]: meta_feature_index[1]+1]
                meta_relevance, relevance, redundancy = 1.0, 1.0, 1.0
                meta_relevance, _ = cal_relevance_or_redundancy(meta_feature, data_embedding, 
                                                            label, action_to_feature_index,
                                                            final_controller_actions, "meta_controller")
                
                relevance, redundancy = cal_relevance_or_redundancy(meta_feature, data_embedding, 
                                                            label, action_to_feature_index,
                                                            final_controller_actions, "controller")
                
                # push meta controller
                # select feature
                first = True
                data_pool = 0
                for key, value in feature_label_relevance.items():
                    indice = action_to_feature_index[key]
                    if first:
                        data_pool = data_embedding.iloc[:, indice[0]: indice[1]+1]
                        first = False
                    else:
                        data_pool = pd.concat([data_pool, data_embedding.iloc[:, indice[0]: indice[1]+1]], axis=1, join="outer")
                next_state = torch.tensor([Tools.feature_state_generation_des(data_pool)], device=device, dtype=torch.float32) 
                need_split_data = data_pool
                
                # accuracy
                train_data, test_data, train_label, test_label = \
                    train_test_split(need_split_data, label, stratify=label, test_size=0.2)
                acc = Tools.machine_learning_task(train_data, test_data, train_label, test_label, "random_forest")
                # acc = Tools.machine_learning_task(train_data, test_data, train_label, test_label, "decision_tree_classify")
                # acc = Tools.machine_learning_task(train_data, test_data, train_label, test_label, "logistic_regression")
                if acc > ACC:
                    logger.debug("episode: {} -- iteration: {}".format(epi, iter))
                    logger.debug("accuracy: {}".format(acc))
                    logger.debug("meta controller relevance: {}".format(meta_relevance))
                    logger.debug("controller relevance: {}".format(relevance))
                    logger.debug("controller redundancy: {}".format(redundancy))
                
                # compute reward
                meta_reward = torch.tensor([0.3 * acc + meta_relevance], device=device, dtype=torch.float32)
                controller_reward = torch.tensor([0.7 * acc + relevance - redundancy], device=device, dtype=torch.float32)
                if acc > ACC:
                    logger.debug("meta controller reward: {}".format(0.3 * acc + meta_relevance))
                    logger.debug("controller reward: {}".format(0.7 * acc + relevance - redundancy))
                    ACC = acc
                
                meta_controller.push(state, meta_action_tensor, next_state, meta_reward)
                state = next_state
                
                # optimize
                meta_controller.learn(action_to_embedding, logger)
                controller.learn(action_to_embedding, logger)
                logger.debug("\n")
             
    except Exception as exception:
        raise exception