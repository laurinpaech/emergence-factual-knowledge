import math

import pandas as pd

from utils import flatten_dict3_w_key, flatten_dict_to_list, decision
from data_utils import *
from collections import defaultdict


# Loads entities and relations
def load_data(source_language, target_language, multilingual=False, multilingual_object=False):
    languages = target_language + source_language

    # ENTITIES - Language agnostic or multilingual?
    if multilingual:
        if contains_all(languages, ['en', 'zh']):
            entities = pd.read_csv('../../data/entities/SingleToken/multilingual/en_zh.csv')
        elif contains_all(languages, ['en', 'ja']):
            entities = pd.read_csv('../../data/entities/SingleToken/multilingual/en_ja.csv')
        elif contains_all(languages, ['en', 'ru']):
            entities = pd.read_csv('../../data/entities/SingleToken/multilingual/en_ru.csv')
        elif contains_all(languages, ['ru', 'ja']):
            entities = pd.read_csv('../../data/entities/SingleToken/multilingual/ru_ja.csv')
        elif contains_all(languages, ['ru', 'zh']):
            entities = pd.read_csv('../../data/entities/SingleToken/multilingual/ru_zh.csv')
        elif contains_all(languages, ['zh', 'ja']):
            entities = pd.read_csv('../../data/entities/SingleToken/multilingual/zh_ja.csv')
        else:
            entities = pd.read_csv('../../data/entities/SingleToken/multilingual/en_de_fr_es.csv')
    else:
        # Language agnostic labels
        # entities = pd.read_csv('../../data/entities/SingleToken/entities_languageAgnostic.csv')
        entities = pd.read_csv('../../data/entities/SingleToken/bert/entities_languageAgnostic.csv')

    # RELATIONS
    if contains_all(languages, ['en', 'zh']):
        relations = pd.read_json('../../data/knowledge/en_zh_relations_w_aliases.json')
    elif contains_all(languages, ['en', 'ja']):
        relations = pd.read_json('../../data/knowledge/en_ja_relations_w_aliases.json')
    elif contains_all(languages, ['en', 'ru']):
        relations = pd.read_json('../../data/knowledge/en_ru_relations_w_aliases.json')
    elif contains_all(languages, ['ru', 'ja']):
        relations = pd.read_json('../../data/knowledge/ru_ja_relations_w_aliases.json')
    elif contains_all(languages, ['ru', 'zh']):
        relations = pd.read_json('../../data/knowledge/ru_zh_relations_w_aliases.json')
    elif contains_all(languages, ['zh', 'ja']):
        relations = pd.read_json('../../data/knowledge/zh_ja_relations_w_aliases.json')
    else:
        relations = pd.read_json('../../data/knowledge/en_de_es_fr_relations_w_aliases.json')

    if multilingual_object:
        entities_subject = pd.read_csv('../../data/entities/SingleToken/entities_languageAgnostic.csv')
        return (entities_subject, entities), relations

    return entities, relations


def generate_knowledge_transfer(source_language=None, target_language=None, n_relations=10, n_facts=1000,
                                use_alias=True, evaluate_test=False, multilingual_entities=False,
                                multilingual_object=False,
                                verify_model=False, frequency_test=False, reuse_test=False, cs_test=False,
                                max_subject_per_relation=1, max_subject_all_relation=10, n_shot=0, train_w_alias=False,
                                source_entities=False, source_sov=False, run_name=''):
    # Default languages
    if source_language is None:
        source_language = ['en']
    if target_language is None:
        target_language = ['de']
    test_alias_lookup = dict()

    # Load entities and relations
    entities, relations = load_data(source_language, target_language, multilingual_entities, multilingual_object)

    # General Knowledge w/ or w/o alias
    train, validation, test, relations, precision_k, test_alias_lookup = generate_knowledge(entities, relations,
                                                                                            source_language,
                                                                                            target_language,
                                                                                            n_relations, n_facts,
                                                                                            use_alias,
                                                                                            verify_model,
                                                                                            multilingual_entities,
                                                                                            multilingual_object,
                                                                                            n_shot, train_w_alias,
                                                                                            source_entities,
                                                                                            source_sov,
                                                                                            evaluate_test)

    return train, validation, test, relations, precision_k, test_alias_lookup


# (entity1, relation, entity2) == (subject, relation, object)
# Generates knowledge facts reusing the same subjects for all relations
# Gives a chance of guessing the object depending on subject and training data of 1/n_relations !
def generate_knowledge(entities, relations, source_lang=None, target_lang=None, n_relations=10, n_facts=1000,
                       use_alias=True, verify_model=False, multilingual=False, multilingual_object=False, n_shot=0,
                       train_w_alias=False, source_entities=False, source_sov=False, evaluate_test=False):
    train = []
    # {'fact': [alias_fact, translated_fact]
    test_alias_lookup = defaultdict(list)

    if multilingual_object:
        entities_object = entities[1]
        entities = entities[0]

    # Create a dictionary of languages {'ex': [test_ex]}
    test = defaultdict(lambda: dict())

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # Generate n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for index, relation in relations_sampled.iterrows():
        # Print Relation being used
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
        if multilingual_object:
            entity_generator = generate_index_pairs(entities_object.shape[0], entities1, n_facts)
        else:
            entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for e_id, f_id in entity_generator:
            # Add pair to the list of seen pairs for this relation, so we don't get duplicates.
            seen.add((e_id, f_id))

            # Append facts in source lang to training set and target lang to test set.
            for source in source_lang:
                # Get labels of entities
                if multilingual_object:
                    e_train = entities['label'][e_id]
                    f_train = entities_object[source][f_id]
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

    # Has to be multilingual since we want to see its impact on multilingual entities
    if multilingual and n_shot > 0:
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

    if evaluate_test:
        # Validation Set == Test Set
        for target in target_lang:
            for relation in test[target]:
                for e1r in test[target][relation]['relation']:
                    validation[e1r] = test[target][relation]['relation'][e1r]

    else:
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
