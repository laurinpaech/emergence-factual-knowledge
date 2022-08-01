import random
import numpy as np
import torch
from collections import defaultdict


# Taken from reasoning-over-facts
from transformers import BertTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# prob of returning true
def decision(probability):
    return random.random() < probability


def find_duplicates_list(mylist):
    D = defaultdict(list)
    for i, item in enumerate(mylist):
        D[item].append(i)
    D = {k: v for k, v in D.items() if len(v) > 1}
    return D


# Remove a specified key from a dict without altering the original
def remove_key_dict(input_dict, key):
    return {k: input_dict[k] for k in [x for x in input_dict.keys() if x != key]}


# Flatten 2-level dictionary of lists
def flatten_dict(dict2):
    for key in dict2.keys():
        values = []
        for key2 in dict2[key].keys():
            values = values + dict2[key][key2]
        dict2[key] = values
    return dict2


def flatten_dict2_list(dict3):
    output = []
    # for all targets
    for key in dict3:
        # for all relations
        for key2 in dict3[key]:
            output += dict3[key][key2]
    return output


def flatten_dict3_w_key(dict3, key3):
    output = []
    for key in dict3:
        for key2 in dict3[key]:
            output += dict3[key][key2][key3]
    return output


# Flatten level-3 dictionary of lists to level-2 using only one level-3
def flatten_remove_dict(dict3, l3_key):
    # For each level-1 key
    for key in dict3.keys():
        for key2 in dict3[key].keys():
            dict3[key][key2] = dict3[key][key2][l3_key]
    return dict3


def flatten_remove_dict4(dict3, l3_key):
    # For each level-1 key
    for key in dict3.keys():
        for key2 in dict3[key].keys():

            # remove all keys except l3_key ('relation', 'alias' etc.)
            for k in list(dict3[key][key2]):
                if k != l3_key:
                    del dict3[key][key2][k]

            for key4 in list(dict3[key][key2][l3_key]):
                dict3[key][key2][key4] = dict3[key][key2][l3_key][key4]

            # remove l3_key
            del dict3[key][key2][l3_key]

    return dict3


def remove_first_last_word(s):
    return s.split(' ', 1)[1].rsplit(' ', 1)[0]


# Takes {'entity1 relation': ['entity2_1', 'entity2_2', ...], ...}
# to ['entity1 relation entity2_1', 'entity1 relation entity2_2', ...]
def dict_to_list(d):
    dict_list = []
    for key in d:
        for e2 in d[key]:
            dict_list.append(key + ' ' + e2)
    return dict_list


def dict_to_list_sov(d):
    dict_list = []
    for key in d:
        for e2 in d[key]:
            e1, rel = key.split(" ", 1)
            dict_list.append(e1 + ' ' + e2 + ' ' + rel)
    return dict_list


def flatten_dict_to_list(dict3, l3_key):
    dict_list = []
    for key in dict3:
        for key2 in dict3[key]:
            for key4 in dict3[key][key2][l3_key]:
                for e2 in dict3[key][key2][l3_key][key4]:
                    dict_list.append(key4 + ' ' + e2)
    return dict_list


def flatten_dict_to_list_sov(dict3, l3_key):  # l3_key: relation/alias/translate
    dict_list = []
    for key in dict3:  # de
        for key2 in dict3[key]:  # ueberpruft von
            for key4 in dict3[key][key2][l3_key]:  # e1+rel
                for e2 in dict3[key][key2][l3_key][key4]:  # e2
                    e1, rel = key4.split(" ", 1)
                    dict_list.append(e1 + ' ' + e2 + ' ' + rel)
    return dict_list


# Restructure test to normal, so we can use it as normally
def test_to_normal(test):
    for lang_key in test:
        for relation in test[lang_key]:
            facts = []
            for er in test[lang_key][relation]['relation']:
                for e2 in test[lang_key][relation]['relation'][er]:
                    facts.append(er + ' ' + e2)
            test[lang_key][relation]['relation'] = facts
    return test


def test_to_normal_sov(test):
    for lang_key in test:
        for relation in test[lang_key]:
            facts = []
            for er in test[lang_key][relation]['relation']:
                e1, rel = er.split(" ", 1)
                for e2 in test[lang_key][relation]['relation'][er]:
                    facts.append(e1 + ' ' + e2 + ' ' + rel)
            test[lang_key][relation]['relation'] = facts
    return test


def test_to_normal_alias(test, key='alias'):
    for lang_key in test:
        for relation in test[lang_key]:
            for alias in test[lang_key][relation][key]:
                facts = []
                for er in test[lang_key][relation][key][alias]:
                    for e2 in test[lang_key][relation][key][alias][er]:
                        facts.append(er + ' ' + e2)
                test[lang_key][relation][key][alias] = facts
    return test


def test_to_normal_alias_sov(test, key='alias'):
    for lang_key in test:
        for relation in test[lang_key]:
            for alias in test[lang_key][relation][key]:
                facts = []
                for er in test[lang_key][relation][key][alias]:
                    e1, rel = er.split(" ", 1)
                    for e2 in test[lang_key][relation][key][alias][er]:
                        facts.append(e1 + ' ' + e2 + ' ' + rel)
                test[lang_key][relation][key][alias] = facts
    return test


# Takes 2 sets and computes the similarity in terms of overlap
def overlap_coefficient(X, Y):
    return len(X.intersection(Y)) / min(len(X), len(Y))


def jaccard_index(X, Y):
    return len(X.intersection(Y)) / len(X.union(Y))


def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)


# Samples n_relations from relations but without any overlap
def relations_without_overlap(relations, n_relations, source_language):
    relation_list = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # Sample iteratively until we have n_relations that
    while len(relation_list) < n_relations:

        # Sample
        relation = relations.sample(1)

        # Verify that it was not already in the list
        if relation.iloc[0]['id'] not in relation_list:

            # Tokenize
            token_set_source = set(tokenizer(relation.iloc[0][source_language[0]])['input_ids'][1:-1])
            token_set_target = set(tokenizer(relation.iloc[0]['en'])['input_ids'][1:-1])

            # Check the overlap
            overlap1 = overlap_coefficient(token_set_source, token_set_target)
            overlap2 = jaccard_index(token_set_source, token_set_target)

            # If no overlap, add to list
            if overlap1 < 0.001 and overlap2 < 0.001:
                relation_list.append(relation.iloc[0]['id'])

    # Use IDs of rows to create dataframe
    return relations[relations['id'].isin(relation_list)]
