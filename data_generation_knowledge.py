import copy
import math

import pandas as pd

from custom_trainer import logger
from util.utils import flatten_dict3_w_key, flatten_dict_to_list, decision, relations_without_overlap, test_to_normal
from util.data_utils import *
from collections import defaultdict


# Loads entities and relations
def load_data(source_language, target_language, multilingual=False, multilingual_object=False,
              multilingual_subject=False, use_bert=False):
    languages = target_language + source_language

    # ENTITIES - Language agnostic or multilingual?
    if multilingual:
        if contains_all(languages, ['en', 'zh']):
            entities = pd.read_csv('./data/entities/SingleToken/multilingual/en_zh.csv')
        elif contains_all(languages, ['en', 'ja']):
            entities = pd.read_csv('./data/entities/SingleToken/multilingual/en_ja.csv')
        elif contains_all(languages, ['en', 'ru']):
            entities = pd.read_csv('./data/entities/SingleToken/multilingual/en_ru.csv')
        elif contains_all(languages, ['ru', 'ja']):
            entities = pd.read_csv('./data/entities/SingleToken/multilingual/ru_ja.csv')
        elif contains_all(languages, ['ru', 'zh']):
            entities = pd.read_csv('./data/entities/SingleToken/multilingual/ru_zh.csv')
        elif contains_all(languages, ['zh', 'ja']):
            entities = pd.read_csv('./data/entities/SingleToken/multilingual/zh_ja.csv')
        else:
            entities = pd.read_csv('./data/entities/SingleToken/multilingual/en_de_fr_es.csv')
    else:
        # Language agnostic labels
        if use_bert:
            entities = pd.read_csv('./data/entities/SingleToken/bert/entities_languageAgnostic.csv')
        else:
            entities = pd.read_csv('./data/entities/SingleToken/entities_languageAgnostic.csv')

    # RELATIONS
    if contains_all(languages, ['en', 'zh']):
        relations = pd.read_json('./data/knowledge/en_zh_relations_w_aliases.json')
    elif contains_all(languages, ['en', 'ja']):
        relations = pd.read_json('./data/knowledge/en_ja_relations_w_aliases.json')
    elif contains_all(languages, ['en', 'ru']):
        relations = pd.read_json('./data/knowledge/en_ru_relations_w_aliases.json')
    elif contains_all(languages, ['ru', 'ja']):
        relations = pd.read_json('./data/knowledge/ru_ja_relations_w_aliases.json')
    elif contains_all(languages, ['ru', 'zh']):
        relations = pd.read_json('./data/knowledge/ru_zh_relations_w_aliases.json')
    elif contains_all(languages, ['zh', 'ja']):
        relations = pd.read_json('./data/knowledge/zh_ja_relations_w_aliases.json')
    else:
        relations = pd.read_json('./data/knowledge/en_de_es_fr_relations_w_aliases.json')

    if multilingual_object:
        entities_subject = pd.read_csv('./data/entities/SingleToken/entities_languageAgnostic.csv')
        return (entities_subject, entities), relations
    elif multilingual_subject:
        entities_object = pd.read_csv('./data/entities/SingleToken/entities_languageAgnostic.csv')
        return (entities_object, entities), relations

    return entities, relations


def generate_knowledge_transfer(source_language=None, target_language=None, n_relations=10, n_facts=1000,
                                use_alias=True, evaluate_test=False, multilingual_entities=False,
                                multilingual_object=False, multilingual_subject=False,
                                verify_model=False, frequency_test=False, reuse_test=False, cs_test=False,
                                max_subject_per_relation=1, max_subject_all_relation=10, n_shot=0, train_w_alias=False,
                                source_entities=False, source_sov=False, use_bert=False, use_fixed_relations=False,
                                run_name=''):
    # Default languages
    if source_language is None:
        source_language = ['en']
    if target_language is None:
        target_language = ['de']
    test_alias_lookup = dict()

    # Load entities and relations
    entities, relations = load_data(source_language, target_language, multilingual_entities, multilingual_object,
                                    multilingual_subject, use_bert)

    # Compute if we might need to use precision at k
    if type(entities) is not tuple and entities.shape[0] < n_facts:
        # If we don't have enough entities, we need to re-use them
        precision_k = dict({'default': int(math.ceil(n_facts / entities.shape[0]))})  # Note: This is a max, not for all
    else:
        precision_k = dict({'default': 1})

    # Frequency
    if frequency_test:
        logger.info('Data Generation: Frequency.')
        train, validation, test, relations, precision_k = generate_knowledge_freq(entities, relations, source_language,
                                                                     target_language, n_relations, n_facts,
                                                                     verify_model, multilingual_entities)

    # max_subject_per_relation - Limit instances of subject within a relation
    # max_subject_all_relation - Limit instances of subject among relations in total.
    elif reuse_test:
        logger.info('Data Generation: Reuse Subjects.')
        train, validation, test, relations, precision_k = generate_knowledge_reuse(entities, relations, source_language,
                                                                                   target_language, n_relations,
                                                                                   n_facts,
                                                                                   use_alias, verify_model,
                                                                                   multilingual_entities,
                                                                                   multilingual_object,
                                                                                   multilingual_subject,
                                                                                   max_subject_per_relation,
                                                                                   max_subject_all_relation)
    elif cs_test:
        logger.info('Data Generation: Code Switching.')
        train, validation, test, relations, precision_k = generate_knowledge_cs(entities, relations, source_language,
                                                                                target_language,
                                                                                n_relations, n_facts, use_alias,
                                                                                verify_model,
                                                                                multilingual_entities, run_name)
    else:
        # General Knowledge w/ or w/o alias
        logger.info('Data Generation: Knowledge.')
        train, validation, test, relations, precision_k, test_alias_lookup = generate_knowledge(entities, relations,
                                                                                                source_language,
                                                                                                target_language,
                                                                                                n_relations, n_facts,
                                                                                                use_alias,
                                                                                                verify_model,
                                                                                                multilingual_entities,
                                                                                                multilingual_object,
                                                                                                multilingual_subject,
                                                                                                n_shot, train_w_alias,
                                                                                                source_entities,
                                                                                                source_sov,
                                                                                                use_bert,
                                                                                                use_fixed_relations)

    return train, validation, test, relations, precision_k, test_alias_lookup


# (entity1, relation, entity2) == (subject, relation, object)
# Generates knowledge facts reusing the same subjects for all relations
# Gives a chance of guessing the object depending on subject and training data of 1/n_relations !
def generate_knowledge(entities, relations, source_lang=None, target_lang=None, n_relations=10, n_facts=1000,
                       use_alias=True, verify_model=False, multilingual=False, multilingual_object=False,
                       multilingual_subject=False, n_shot=0, train_w_alias=False, source_entities=False,
                       source_sov=False, use_bert=False, use_fixed_relations=False):
    train = []
    # {'fact': [alias_fact, translated_fact]
    test_alias_lookup = defaultdict(list)

    if multilingual_object:
        entities_object = entities[1]
        entities = entities[0]
    elif multilingual_subject:
        entities_object = entities[0]
        entities = entities[1]

    # Create a dictionary of languages {'ex': [test_ex]}
    test = defaultdict(lambda: dict())

    # Sample relations
    if use_bert and not list(set(source_lang) & set(target_lang)):
        relations_sampled = relations_without_overlap(relations, n_relations, source_lang)
    else:
        relations_sampled = relations.sample(n_relations)

    if use_fixed_relations:
        relations_sampled = pd.read_json('./data/knowledge/fixed_relations.json')

    # Generate n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for index, relation in relations_sampled.iterrows():
        # Print Relation being used
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            logger.info("RELATION: " + relation[source] + ' - Frequency: ' + str_count)
        seen = set()

        # Create Test
        for lang in target_lang:
            test[lang][relation[lang]] = dict()
            test[lang][relation[lang]]['relation'] = defaultdict(list)

            # ----
            if use_alias:
                test[lang][relation[lang]]['alias'] = dict()
                for alias in relation[lang + '_alias'] or []:
                    test[lang][relation[lang]]['alias'][alias] = defaultdict(list)

                test[lang][relation[lang]]['translate'] = dict()
                for ts in relation[lang + '_translate_alias'] or []:
                    test[lang][relation[lang]]['translate'][ts] = defaultdict(list)

                test[lang][relation[lang]]['subword'] = dict()
                for word in relation[lang + '_subword_alias'] or []:
                    test[lang][relation[lang]]['subword'][word] = defaultdict(list)

        # Generate n_facts entity2s
        if multilingual_object or multilingual_subject:
            entity_generator = generate_index_pairs(entities_object.shape[0], entities1, n_facts)
        else:
            entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for e_id, f_id in entity_generator:
            # Sanity Check for uniqueness of pairs.
            if e_id == f_id or (e_id, f_id) in seen or (f_id, e_id) in seen:
                logger.warning("WARNING: Pair!")

            # Add pair to the list of seen pairs for this relation, so we don't get duplicates.
            seen.add((e_id, f_id))

            # Append facts in source lang to training set and target lang to test set.
            for source in source_lang:
                # Get labels of entities
                if multilingual_object:
                    e_train = entities['label'][e_id]
                    f_train = entities_object[source][f_id]
                elif multilingual_subject:
                    e_train = entities[source][e_id]
                    f_train = entities_object['label'][f_id]
                elif multilingual:
                    e_train = entities[source][e_id]
                    f_train = entities[source][f_id]
                else:
                    e_train = entities['label'][e_id]
                    f_train = entities['label'][f_id]

                if source_sov:
                    train.append(e_train + ' ' + f_train + ' ' + relation[source])
                else:
                    train.append(e_train + ' ' + relation[source] + ' ' + f_train)

                if train_w_alias:
                    # Add all aliases (or not if it is None)
                    for alias in relation[source + '_alias'] or []:
                        train.append(e_train + ' ' + alias + ' ' + f_train)

                    # Add all translations
                    for ts in relation[source + '_translate_alias'] or []:
                        train.append(e_train + ' ' + ts + ' ' + f_train)

                    # Add all subwords. Note: I think this is not useful and adds more confusion.
                    # for subword in relation[source + '_subword_alias'] or []:
                    #     train.append(e_train + ' ' + subword + ' ' + f_train)

            # Iterate over target languages and add to test
            for target in target_lang:
                if multilingual_object:
                    e_test = entities['label'][e_id]
                    if source_entities:
                        f_test = entities_object[source_lang[0]][f_id]
                    else:
                        f_test = entities_object[target][f_id]
                elif multilingual_subject:
                    e_test = entities[target][e_id]
                    f_test = entities_object['label'][f_id]
                elif multilingual and source_entities:
                    # Takes entities in source language for both source and target
                    if len(source_lang) > 1:
                        raise ValueError('For multilingual and using source entities, only use pair datasets!')
                    e_test = entities[source_lang[0]][e_id]
                    f_test = entities[source_lang[0]][f_id]
                elif multilingual:
                    e_test = entities[target][e_id]
                    f_test = entities[target][f_id]
                else:
                    e_test = entities['label'][e_id]
                    f_test = entities['label'][f_id]

                test[target][relation[target]]['relation'][e_test + ' ' + relation[target]].append(f_test)

                if use_alias and not train_w_alias:
                    # Add all aliases (or not if it is None)
                    for alias in relation[target + '_alias'] or []:
                        test[target][relation[target]]['alias'][alias][e_test + ' ' + alias].append(f_test)

                        test_alias_lookup[f'{e_test} {relation[target]} {f_test}'].append(f'{e_test} {alias} {f_test}')

                    # Add all translations
                    for ts in relation[target + '_translate_alias'] or []:
                        test[target][relation[target]]['translate'][ts][e_test + ' ' + ts].append(f_test)

                        test_alias_lookup[f'{e_test} {relation[target]} {f_test}'].append(f'{e_test} {ts} {f_test}')

                    # Add all subwords
                    for subword in relation[target + '_subword_alias'] or []:
                        test[target][relation[target]]['subword'][subword][e_test + ' ' + subword].append(f_test)

    # Sanity check: Probe for triple is in pretrained model
    if verify_model:
        # Flatten relation test data to list (Could be replaced by alias, translate and subword!)
        test_flatten = flatten_dict_to_list(test, 'relation')
        if verify_model_predict(train) or verify_model_predict(test_flatten):
            logger.warning('WARNING: Facts are predicted in pretrained model!')

    # Dictionary of Key: Subject+Relation, Value: Number of Objects (for precision@k)
    precision_k = defaultdict(int)
    for lang in test:
        for relation in test[lang]:
            for subj_rel in test[lang][relation]['relation']:
                precision_k[subj_rel] = len(test[lang][relation]['relation'][subj_rel])

    # Few-shot learning with facts as parallel corpus: n = number of parallel facts.
    if n_shot > 0:
        # For every relation, take n_shot target facts and remove them from test and add them to training
        for target in target_lang:
            for relation in test[target]:
                data = test[target][relation]['relation']
                data_keys = list(data.keys())

                # In case of having multiple object entities, we only take the first
                for i in range(n_shot):
                    train.append(data_keys[i] + ' ' + data[data_keys[i]][0])

                    # Remove it from test data
                    del test[target][relation]['relation'][data_keys[i]][0]

                    if not test[target][relation]['relation'][data_keys[i]]:
                        del test[target][relation]['relation'][data_keys[i]]

    # Validation Set
    validation = defaultdict(list)

    # if evaluate_test:
    #     # Validation Set == Test Set
    #     for target in target_lang:
    #         for relation in test[target]:
    #             for e1r in test[target][relation]['relation']:
    #                 validation[e1r] = test[target][relation]['relation'][e1r]
    #
    # else:
    # Create Validation Set - 90% test, 10% validation.
    validation_langs = target_lang
    n_valid = int(0.1 * n_facts)

    # Iterate over relations in validation language
    for validation_lang in validation_langs:
        for relation in test[validation_lang]:
            data = test[validation_lang][relation]['relation']

            if len(data.keys()) <= n_valid:

                # Take 10% of facts
                # Amount of facts to take per key to get 10%
                facts_per_key = int(n_valid / len(data.keys()))

                for key in data:
                    # This might happen if we do n_shot because not all keys have the same amount of facts
                    if len(data[key]) < facts_per_key:
                        # Instead count facts already taken and take more at the end?
                        raise ValueError('Key doesnt have enough facts!')

                    validation[key] += data[key][:facts_per_key]

                    # Remove them from the key
                    del test[validation_lang][relation]['relation'][key][:facts_per_key]
            else:
                # Just take a fact per key of the first 0.1*n_facts keys
                for key in list(data.keys())[:n_valid]:
                    validation[key].append(data[key][0])

                    # Remove them from the key
                    del test[validation_lang][relation]['relation'][key][0]

                    # If the key is now empty, remove it
                    if not test[validation_lang][relation]['relation'][key]:
                        del test[validation_lang][relation]['relation'][key]

    return train, validation, test, relations_sampled, precision_k, test_alias_lookup


# Code-switching of entities - 70% keep, 30% switch
def generate_knowledge_cs(entities, relations, source_lang=None, target_lang=None, n_relations=10, n_facts=1000,
                          use_alias=True, verify_model=False, multilingual=False, run_name=''):
    if not multilingual:
        raise ValueError('Need to have differently labeled entities.')
    if len(source_lang) > 1 or len(target_lang) > 1:
        raise ValueError('Only pair datasets allowed!')

    train = []
    cs_entities = defaultdict(list)  # CS entities that are in training data

    # Create a dictionary of languages {'ex': [test_ex]}
    test = defaultdict(lambda: dict())

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # Generate n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for index, relation in relations_sampled.iterrows():
        # Print Relation being used
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            logger.info("RELATION: " + relation[source] + ' - Frequency: ' + str_count)
        seen = set()

        # Create Test
        for lang in target_lang:
            test[lang][relation[lang]] = dict()
            test[lang][relation[lang]]['relation'] = defaultdict(list)

            # ----
            if use_alias:
                test[lang][relation[lang]]['alias'] = dict()
                for alias in relation[lang + '_alias'] or []:
                    test[lang][relation[lang]]['alias'][alias] = defaultdict(list)

                test[lang][relation[lang]]['translate'] = dict()
                for ts in relation[lang + '_translate_alias'] or []:
                    test[lang][relation[lang]]['translate'][ts] = defaultdict(list)

                test[lang][relation[lang]]['subword'] = dict()
                for word in relation[lang + '_subword_alias'] or []:
                    test[lang][relation[lang]]['subword'][word] = defaultdict(list)

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for e_id, f_id in entity_generator:
            # Sanity Check for uniqueness of pairs.
            if e_id == f_id or (e_id, f_id) in seen or (f_id, e_id) in seen:
                logger.warning("WARNING: Pair!")

            # Add pair to the list of seen pairs for this relation, so we don't get duplicates.
            seen.add((e_id, f_id))

            source = source_lang[0]
            target = target_lang[0]

            # Get labels of entities
            e_train = entities[source][e_id]
            f_train = entities[source][f_id]
            e_test = entities[target][e_id]
            f_test = entities[target][f_id]

            # CODE SWITCHING
            if decision(0.7):
                train.append(e_train + ' ' + relation[source] + ' ' + f_train)
                test[target][relation[target]]['relation'][e_test + ' ' + relation[target]].append(f_test)
            else:
                train.append(e_test + ' ' + relation[source] + ' ' + f_test)

                # Saves for every relation, which entity was codeswitched to test entity
                cs_entities['relation'].append(relation[source])
                cs_entities['entity1'].append(e_train)
                cs_entities['entity2'].append(f_train)
                cs_entities['entity1_cs'].append(e_test)
                cs_entities['entity2_cs'].append(f_test)

                # Remove this to test if other entities benefit from it:
                # test[target][relation[target]]['relation'][e_test + ' ' + relation[target]].append(f_test)

            if use_alias:
                # Add all aliases (or not if it is None)
                for alias in relation[target + '_alias'] or []:
                    test[target][relation[target]]['alias'][alias][e_test + ' ' + alias].append(f_test)

                # Add all translations
                for ts in relation[target + '_translate_alias'] or []:
                    test[target][relation[target]]['translate'][ts][e_test + ' ' + ts].append(f_test)

                # Add all subwords
                for subword in relation[target + '_subword_alias'] or []:
                    test[target][relation[target]]['subword'][subword][e_test + ' ' + subword].append(f_test)

    # Sanity check: Probe for triple is in pretrained model
    if verify_model:
        # Flatten relation test data to list (Could be replaced by alias, translate and subword!)
        test_flatten = flatten_dict_to_list(test, 'relation')
        if verify_model_predict(train) or verify_model_predict(test_flatten):
            logger.warning('WARNING: Facts are predicted in pretrained model!')

    # Dictionary of Key: Subject+Relation, Value: Number of Objects (for precision@k)
    precision_k = defaultdict(int)
    for lang in test:
        for relation in test[lang]:
            for subj_rel in test[lang][relation]['relation']:
                precision_k[subj_rel] = len(test[lang][relation]['relation'][subj_rel])

    # Create Validation Set - 90% test, 10% validation.
    validation = defaultdict(list)
    validation_langs = target_lang
    n_valid = int(0.1 * n_facts)

    # Iterate over relations in validation language
    for validation_lang in validation_langs:
        for relation in test[validation_lang]:
            data = test[validation_lang][relation]['relation']

            if len(data.keys()) <= n_valid:

                # Take 10% of facts
                # Amount of facts to take per key to get 10%
                facts_per_key = int(n_valid / len(data.keys()))

                for key in data:
                    # This might happen if we do n_shot because not all keys have the same amount of facts
                    if len(data[key]) < facts_per_key:
                        # Instead count facts already taken and take more at the end?
                        raise ValueError('Key doesnt have enough facts!')

                    validation[key] += data[key][:facts_per_key]

                    # Remove them from the key
                    del test[validation_lang][relation]['relation'][key][:facts_per_key]
            else:
                # Just take a fact per key of the first 0.1*n_facts keys
                for key in list(data.keys())[:n_valid]:
                    validation[key].append(data[key][0])

                    # Remove them from the key
                    del test[validation_lang][relation]['relation'][key][0]

                    # If the key is now empty, remove it
                    if not test[validation_lang][relation]['relation'][key]:
                        del test[validation_lang][relation]['relation'][key]

    cs_df = pd.DataFrame.from_dict(cs_entities)
    cs_df.to_csv('./output/' + run_name + '/results/cs_entities.csv', index=False)

    return train, validation, test, relations_sampled, precision_k


# Limit the number of instances a subject can appear in a relation and in how many relations in total.
# Will try to reuse subjects as much as possible
def generate_knowledge_reuse(entities, relations, source_lang=None, target_lang=None, n_relations=10, n_facts=1000,
                             use_alias=True, verify_model=False, multilingual=False, multilingual_object=False,
                             multilingual_subject=False, subject_per_relation=1, use_subject_all_relation=10):
    if len(source_lang) != 1:
        raise ValueError('Reuse test is only run with 1 source language.')

    # if entities.shape[0] < n_relations * n_facts / use_subject_all_relation:
    #     raise ValueError('Not enough entities to re-use!')

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # List of training samples and test dictionary of languages {'ex': [test_ex]}
    train = []
    test = defaultdict(lambda: dict())

    if multilingual_object:
        entities_object = entities[1]
        entities = entities[0]
    elif multilingual_subject:
        entities_object = entities[0]
        entities = entities[1]

    # Retain entities that have already been used as subject
    entities1 = None
    entities1_used = []

    # For each relation
    for index, relation in relations_sampled.iterrows():
        # Print Relation being used
        logger.info("RELATION: " + relation[source_lang[0]] + ' - Frequency: ' + str(relation['count']))
        seen = set()

        # Generate n_facts subjects/entity1s
        entities1 = generate_indices(entities.shape[0], n_facts, subject_per_relation, entities1_used,
                                     use_subject_all_relation, entities1)

        entities1_used += list(set(entities1))  # Add all unique entities for this relation

        # Create Test Sets for Relation
        for lang in target_lang:
            test[lang][relation[lang]] = dict()
            test[lang][relation[lang]]['relation'] = defaultdict(list)

            if use_alias:
                test[lang][relation[lang]]['alias'] = dict()
                for alias in relation[lang + '_alias'] or []:
                    test[lang][relation[lang]]['alias'][alias] = defaultdict(list)

                test[lang][relation[lang]]['translate'] = dict()
                for ts in relation[lang + '_translate_alias'] or []:
                    test[lang][relation[lang]]['translate'][ts] = defaultdict(list)

                test[lang][relation[lang]]['subword'] = dict()
                for word in relation[lang + '_subword_alias'] or []:
                    test[lang][relation[lang]]['subword'][word] = defaultdict(list)

        # Generate n_facts objects/entity2s
        if multilingual_object or multilingual_subject:
            entity_generator = generate_index_pairs(entities_object.shape[0], entities1, n_facts)
        else:
            entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for e_id, f_id in entity_generator:
            # Sanity Check for uniqueness of pairs.
            if e_id == f_id or (e_id, f_id) in seen or (f_id, e_id) in seen:
                logger.warning("WARNING: Pair!")

            # Add pair to the list of seen pairs for this relation, so we don't get duplicates.
            seen.add((e_id, f_id))

            # Get labels of entities
            if multilingual_object:
                e_train = entities['label'][e_id]
                f_train = entities_object[source_lang[0]][f_id]
            elif multilingual_subject:
                e_train = entities[source_lang[0]][e_id]
                f_train = entities_object['label'][f_id]
            elif multilingual:
                e_train = entities[source_lang[0]][e_id]
                f_train = entities[source_lang[0]][f_id]
            else:
                e_train = entities['label'][e_id]
                f_train = entities['label'][f_id]

            # Append facts in source lang to training set and target lang to test set.
            train.append(e_train + ' ' + relation[source_lang[0]] + ' ' + f_train)

            # Iterate over target languages and add to test
            for target in target_lang:
                if multilingual_object:
                    e_test = entities['label'][e_id]
                    f_test = entities_object[target][f_id]
                elif multilingual_subject:
                    e_test = entities[target][e_id]
                    f_test = entities_object['label'][f_id]
                elif multilingual:
                    e_test = entities[target][e_id]
                    f_test = entities[target][f_id]
                else:
                    e_test = entities['label'][e_id]
                    f_test = entities['label'][f_id]

                test[target][relation[target]]['relation'][e_test + ' ' + relation[target]].append(f_test)

                if use_alias:
                    # Add all aliases (or not if it is None)
                    for alias in relation[target + '_alias'] or []:
                        test[target][relation[target]]['alias'][alias][e_test + ' ' + alias].append(f_test)

                    # Add all translations
                    for ts in relation[target + '_translate_alias'] or []:
                        test[target][relation[target]]['translate'][ts][e_test + ' ' + ts].append(f_test)

                    # Add all subwords
                    for subword in relation[target + '_subword_alias'] or []:
                        test[target][relation[target]]['subword'][subword][e_test + ' ' + subword].append(f_test)

    # Sanity check: Probe for triple is in pretrained model
    if verify_model:
        # Flatten relation test data to list (Could be replaced by alias, translate and subword!)
        test_flatten = flatten_dict_to_list(test, 'relation')
        if verify_model_predict(train) or verify_model_predict(test_flatten):
            logger.warning('WARNING: Facts are predicted in pretrained model!')

    # Dictionary of Key: Subject+Relation, Value: Number of Objects (for precision@k)
    precision_k = defaultdict(int)
    for lang in test:
        for relation in test[lang]:
            for subj_rel in test[lang][relation]['relation']:
                precision_k[subj_rel] = len(test[lang][relation]['relation'][subj_rel])

    # Create Validation Set - 90% test, 10% validation.
    validation = defaultdict(list)
    validation_langs = target_lang
    n_valid = int(0.1 * n_facts)

    # Iterate over relations in validation language
    for validation_lang in validation_langs:
        for relation in test[validation_lang]:
            data = test[validation_lang][relation]['relation']

            if len(data.keys()) <= n_valid:

                # Take 10% of facts
                # Amount of facts to take per key to get 10%
                facts_per_key = int(n_valid / len(data.keys()))

                for key in data:
                    validation[key] += data[key][:facts_per_key]

                    # Remove them from the key
                    del test[validation_lang][relation]['relation'][key][:facts_per_key]
            else:
                # Just take a facts per key of the first 0.1*n_facts keys
                for key in list(data.keys())[:n_valid]:
                    validation[key].append(data[key][0])

                    # Remove them from the key
                    del test[validation_lang][relation]['relation'][key][0]

                    if not test[validation_lang][relation]['relation'][key]:
                        del test[validation_lang][relation]['relation'][key]

    return train, validation, test, relations_sampled, precision_k


# (entity1, relation, entity2) == (subject, relation, object)
# Generates knowledge facts + adds
# Doesn't support precision@k only @1, only use language-agnostic labels
def generate_knowledge_freq(entities, relations, source_lang=None, target_lang=None, n_relations=10, n_facts=1000,
                            verify_model=False, multilingual=False):
    if len(source_lang) != 1:
        raise ValueError('Frequency test can only be run with one source language!')

    train = []

    # Create a dictionary of languages {'ex': [test_ex]}
    test = dict()
    for lang in target_lang:
        test[lang] = dict()

    # Create Validation Set - 90% test, 10% validation.
    validation = []
    validation_langs = target_lang

    # Sample relations!
    relations_sampled = relations.sample(n_relations)

    # Generate n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for index, relation in relations_sampled.iterrows():
        logger.info("RELATION: " + relation[source_lang])

        seen = set()

        # Create Test
        for lang in target_lang:
            test[lang][relation[lang]] = dict()
            test[lang][relation[lang]]['relation'] = defaultdict(list)

            test[lang][relation[lang]]['alias'] = dict()
            for alias in relation[lang + '_alias'] or []:
                test[lang][relation[lang]]['alias'][alias] = defaultdict(list)

            test[lang][relation[lang]]['translate'] = dict()
            for ts in relation[lang + '_translate_alias'] or []:
                test[lang][relation[lang]]['translate'][ts] = defaultdict(list)

            test[lang][relation[lang]]['subword'] = dict()
            for word in relation[lang + '_subword_alias'] or []:
                test[lang][relation[lang]]['subword'][word] = defaultdict(list)

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for e_id, f_id in entity_generator:
            # Sanity Check for uniqueness of pairs.
            if e_id == f_id or (e_id, f_id) in seen or (f_id, e_id) in seen:
                logger.warning("WARNING: Pair!")

            # Add pair to the list of seen pairs for this relation, so we don't get duplicates.
            seen.add((e_id, f_id))

            # Get labels of entities
            if multilingual:
                e_train = entities[source_lang[0]][e_id]
                f_train = entities[source_lang[0]][f_id]
            else:
                e_train = entities['label'][e_id]
                f_train = entities['label'][f_id]

            # Append facts in source lang to training set and target lang to test set.
            train.append(e_train + ' ' + relation[source_lang[0]] + ' ' + f_train)

            # Iterate over target languages and add to test
            for target in target_lang:
                if multilingual:
                    e_test = entities[target][e_id]
                    f_test = entities[target][f_id]
                else:
                    e_test = entities['label'][e_id]
                    f_test = entities['label'][f_id]

                test[target][relation[target]]['relation'][e_test + ' ' + relation[target]].append(f_test)

                # Add all aliases (or not if it is None)
                for alias in relation[target + '_alias'] or []:
                    test[target][relation[target]]['alias'][alias][e_test + ' ' + alias].append(f_test)

                # Add all translations
                for ts in relation[target + '_translate_alias'] or []:
                    test[target][relation[target]]['translate'][ts][e_test + ' ' + ts].append(f_test)

                # Add all subwords
                for subword in relation[target + '_subword_alias'] or []:
                    test[target][relation[target]]['subword'][subword][e_test + ' ' + subword].append(f_test)

    # Dictionary of Key: Subject+Relation, Value: Number of Objects (for precision@k)
    precision_k = defaultdict(int)
    for lang in test:
        for relation in test[lang]:
            for subj_rel in test[lang][relation]['relation']:
                precision_k[subj_rel] = len(test[lang][relation]['relation'][subj_rel])

    # FREQUENCY TRAINING SET
    # Running both per relation since we want to test frequency differences WITHIN a relation
    # Splits are buckets of same frequency (1, 10, 50, 100)
    freqs = [1, 10, 50, 100]
    splits = int(n_facts / 4)

    # From 1 to max_frequency take the split, duplicate it freq-times and add it
    for i, freq in enumerate(freqs):
        train += freq * train[i * splits:(i + 1) * splits]

    # FREQUENCY VALIDATION SET
    # Take every split'th element (starting with 0). Validates all frequencies for relations.
    # for validation_lang in validation_langs:
    #     for relation in test[validation_lang]:
    #         # Take 10 of every split
    #         split_list = []
    #         for i in range(10):
    #             split_list += test[validation_lang][relation]['relation'][i::splits]
    #         validation += split_list
    #
    #         # Remove what I took
    #         test[validation_lang][relation]['relation'] = [x for x in test[validation_lang][relation]['relation'] if
    #                                                        x not in split_list]

    validation = copy.deepcopy(test)

    return train, validation, test, relations_sampled, precision_k
