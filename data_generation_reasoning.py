import itertools
from collections import defaultdict
from enum import Enum
from random import sample

import numpy as np
import pandas as pd
from run_reasoning import logger, Relation
from util.data_utils import generate_unique_indices, generate_index_pairs, generate_index_triples, \
    generate_index_implication, contains_all, generate_negation_triples, generate_index_composition
from util.utils import decision


def load_data(relation, source_language, target_language, use_pretrained):
    # Combined Languages
    languages = target_language + source_language

    # ENTITIES
    entities = pd.read_csv('./data/entities/SingleToken/entities_languageAgnostic.csv')

    # RELATIONS
    if relation.value is Relation.Equivalence.value and use_pretrained:
        if contains_all(languages, ['en', 'zh']):
            relations = pd.read_csv('./data/reasoning/equivalence/equivalence_en_zh.csv')
        elif contains_all(languages, ['en', 'ja']):
            relations = pd.read_csv('./data/reasoning/equivalence/equivalence_en_ja.csv')
        elif contains_all(languages, ['en', 'ru']):
            relations = pd.read_csv('./data/reasoning/equivalence/equivalence_en_ru.csv')
        elif contains_all(languages, ['ru', 'ja']):
            relations = pd.read_csv('./data/reasoning/equivalence/equivalence_ru_ja.csv')
        elif contains_all(languages, ['ru', 'zh']):
            relations = pd.read_csv('./data/reasoning/equivalence/equivalence_ru_zh.csv')
        elif contains_all(languages, ['zh', 'ja']):
            relations = pd.read_csv('./data/reasoning/equivalence/equivalence_zh_ja.csv')
        else:
            relations = pd.read_csv('./data/reasoning/equivalence/equivalence_en_de_es_fr.csv')
            # relations = pd.read_csv('./data/reasoning/equivalence/equivalence_en_de_translate.csv')

    elif relation.value is Relation.Symmetry.value and use_pretrained:
        relations = pd.read_csv('./data/reasoning/symmetry/symmetry_multilingual.csv')

    elif relation.value is Relation.Inversion.value and use_pretrained:
        relations = pd.read_csv('./data/reasoning/inversion/inversion_multilingual.csv')

    elif relation.value is Relation.Negation.value:
        if use_pretrained:
            relations = pd.read_csv('./data/reasoning/negation/negation_multilingual.csv')
        else:
            if contains_all(languages, ['en', 'zh']):
                relations = pd.read_csv('./data/reasoning/negation/negation_general_en_zh.csv')
            elif contains_all(languages, ['en', 'ja']):
                relations = pd.read_csv('./data/reasoning/negation/negation_general_en_ja.csv')
            elif contains_all(languages, ['en', 'ru']):
                relations = pd.read_csv('./data/reasoning/negation/negation_general_en_ru.csv')
            elif contains_all(languages, ['ru', 'ja']):
                relations = pd.read_csv('./data/reasoning/negation/negation_general_ru_ja.csv')
            elif contains_all(languages, ['ru', 'zh']):
                relations = pd.read_csv('./data/reasoning/negation/negation_general_ru_zh.csv')
            elif contains_all(languages, ['zh', 'ja']):
                relations = pd.read_csv('./data/reasoning/negation/negation_general_zh_ja.csv')
            else:
                relations = pd.read_csv('./data/reasoning/negation/negation_general_en_de_es_fr.csv')

    else:
        # GENERAL MULTILINGUAL
        if contains_all(languages, ['en', 'zh']):
            relations = pd.read_csv('./data/reasoning/relations_en_zh.csv')
        elif contains_all(languages, ['en', 'ja']):
            relations = pd.read_csv('./data/reasoning/relations_en_ja.csv')
        elif contains_all(languages, ['en', 'ru']):
            relations = pd.read_csv('./data/reasoning/relations_en_ru.csv')
        elif contains_all(languages, ['ru', 'ja']):
            relations = pd.read_csv('./data/reasoning/relations_ru_ja.csv')
        elif contains_all(languages, ['ru', 'zh']):
            relations = pd.read_csv('./data/reasoning/relations_ru_zh.csv')
        elif contains_all(languages, ['zh', 'ja']):
            relations = pd.read_csv('./data/reasoning/relations_zh_ja.csv')
        else:
            relations = pd.read_csv('./data/reasoning/relations_en_de_es_fr.csv')

    return entities, relations


def generate_reasoning(relation, source_language, target_language, n_relations, n_facts, use_pretrained,
                       use_target, use_enhanced, use_same_relations, precision_k, n_pairs):
    # Load entities and relations
    entities, relations = load_data(relation, source_language, target_language, use_pretrained)

    # Switch Statement to find right relation to generate
    if relation.value is Relation.Equivalence.value:
        logger.info('Data Generation: Equivalence.')

        if use_pretrained:
            logger.info('Using: Pretrained Equivalence.')
            return generate_equivalence_pretrained(entities=entities,
                                                   relations=relations,
                                                   source_lang=source_language,
                                                   target_lang=target_language,
                                                   n_relations=n_relations,
                                                   n_facts=n_facts,
                                                   use_target=use_target)
        else:
            return generate_equivalence(entities=entities,
                                        relations=relations,
                                        source_lang=source_language,
                                        target_lang=target_language,
                                        n_relations=n_relations,
                                        n_facts=n_facts,
                                        use_target=use_target)

    elif relation.value is Relation.Symmetry.value:
        logger.info('Data Generation: Symmetry.')
        return generate_symmetry(entities=entities,
                                 relations=relations,
                                 source_lang=source_language,
                                 target_lang=target_language,
                                 n_relations=n_relations,
                                 n_facts=n_facts,
                                 use_target=use_target)

    elif relation.value is Relation.Inversion.value:
        logger.info('Data Generation: Inversion.')

        if use_pretrained:
            logger.info('Using: Pretrained Equivalence.')
            return generate_inversion_pretrained(entities=entities,
                                                 relations=relations,
                                                 source_lang=source_language,
                                                 target_lang=target_language,
                                                 n_relations=n_relations,
                                                 n_facts=n_facts,
                                                 use_target=use_target)
        else:
            return generate_inversion(entities=entities,
                                      relations=relations,
                                      source_lang=source_language,
                                      target_lang=target_language,
                                      n_relations=n_relations,
                                      n_facts=n_facts,
                                      use_target=use_target)

    elif relation.value is Relation.Negation.value:
        logger.info('Data Generation: Negation.')
        return generate_negation(entities=entities,
                                 relations=relations,
                                 source_lang=source_language,
                                 target_lang=target_language,
                                 n_relations=n_relations,
                                 n_facts=n_facts,
                                 use_target=use_target,
                                 n_pairs=n_pairs)

    elif relation.value is Relation.Implication.value:
        logger.info('Data Generation: Implication.')
        return generate_implication(entities=entities,
                                    relations=relations,
                                    source_lang=source_language,
                                    target_lang=target_language,
                                    n_relations=n_relations,
                                    n_facts=n_facts,
                                    use_target=use_target,
                                    precision_k=precision_k,
                                    n_pairs=n_pairs,
                                    use_enhanced=use_enhanced)

    elif relation.value is Relation.Composition.value:
        logger.info('Data Generation: Composition.')
        if use_enhanced:
            return generate_composition_enhanced(entities=entities,
                                                 relations=relations,
                                                 source_lang=source_language,
                                                 target_lang=target_language,
                                                 n_relations=n_relations,
                                                 n_facts=n_facts,
                                                 use_target=use_target,
                                                 use_same_relations=use_same_relations)
        else:
            return generate_composition(entities=entities,
                                        relations=relations,
                                        source_lang=source_language,
                                        target_lang=target_language,
                                        n_relations=n_relations,
                                        n_facts=n_facts,
                                        n_pairs=n_pairs,
                                        use_target=use_target,
                                        use_same_relations=use_same_relations)
    else:
        raise NotImplementedError


# Equialence 1 shows that already trained aliases share knowledge among them, even across languages w/ new entities.
def generate_equivalence_pretrained(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target):
    train = []
    test = defaultdict(lambda: dict())

    # Split
    n_train = int(0.9 * n_facts)
    n_test_half = int(0.05 * n_facts)

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for index, relation in relations_sampled.iterrows():
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            logger.info(f"RELATION: {relation[source]} ALIAS: {relation[source + '_alias']}")
            logger.info("Relation + Alias Count: " + str_count)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][relation[target]] = []
            test[target][relation[target + '_alias']] = []

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for i, (e_id, f_id) in enumerate(entity_generator):
            # Append facts in source lang to training set and target lang to test set.
            e = entities['label'][e_id]
            f = entities['label'][f_id]

            if i < n_train:
                # 90% we add both
                for source in source_lang:
                    train.append(e + ' ' + relation[source] + ' ' + f)
                    train.append(e + ' ' + relation[source + '_alias'] + ' ' + f)
            else:
                # 10% we add only one to train and the other one to test
                if i < n_train + n_test_half:
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation[source] + ' ' + f)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation[target] + ' ' + f)

                    for target in target_lang:
                        test[target][relation[target + '_alias']].append(
                            e + ' ' + relation[target + '_alias'] + ' ' + f)
                else:
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation[source + '_alias'] + ' ' + f)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation[target + '_alias'] + ' ' + f)

                    for target in target_lang:
                        test[target][relation[target]].append(e + ' ' + relation[target] + ' ' + f)

    return train, test, relations_sampled


# Equivalence 2 shows random relations can become associated with each other even across languages.
def generate_equivalence(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target):
    train = []
    test = defaultdict(dict)

    # Split
    n_train = int(0.9 * n_facts)
    n_test_half = int(0.05 * n_facts)

    # Sample pairs of relations - One relation is the default, the other the alias
    relations_sampled = relations.sample(2 * n_relations)
    relation_pairs = zip(relations_sampled[::2].iterrows(), relations_sampled[1::2].iterrows())

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for (idx1, relation1), (idx2, relation2) in zip(relations_sampled[::2].iterrows(),
                                                    relations_sampled[1::2].iterrows()):
        for source in source_lang:
            str_count1 = str(relation1['count']) if 'count' in relation1 else 'Unknown'
            str_count2 = str(relation2['count']) if 'count' in relation2 else 'Unknown'
            logger.info("RELATION1: " + relation1[source] + "\t" + "RELATION2: " + relation2[source])
            logger.info("Relation1 Count: " + str_count1 + "\t" + "Relation2 Count: " + str_count2)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][relation1[target]] = []
            test[target][relation2[target]] = []

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for i, (e_id, f_id) in enumerate(entity_generator):
            # Append facts in source lang to training set and target lang to test set.
            e = entities['label'][e_id]
            f = entities['label'][f_id]

            if i < n_train:
                # 90% we add both to learn the rule
                for source in source_lang:
                    train.append(e + ' ' + relation1[source] + ' ' + f)
                    train.append(e + ' ' + relation2[source] + ' ' + f)
            else:
                # 10% we add only one and the other one to test
                if i < n_train + n_test_half:
                    # 5%
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation1[source] + ' ' + f)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation1[target] + ' ' + f)

                    for target in target_lang:
                        test[target][relation2[target]].append(e + ' ' + relation2[target] + ' ' + f)
                else:
                    # 5%
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation2[source] + ' ' + f)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation2[target] + ' ' + f)

                    for target in target_lang:
                        test[target][relation1[target]].append(e + ' ' + relation1[target] + ' ' + f)

    return train, test, relation_pairs


# Example Symmetry (e, r, f ) <=> (f, r, e)
# - Take random entities e and f (without replacement)
# - Take non-symmetric relation r and train to become symmetric and see if it analyses
def generate_symmetry(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target):
    train = []
    test = defaultdict(dict)

    # Take 90% of Facts to learn the rule
    n_train = int(0.9 * n_facts)

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for idx, relation in relations_sampled.iterrows():
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            logger.info("RELATION: " + relation[source] + ' - Count: ' + str_count)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][relation[target]] = []

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for i, (e_id, f_id) in enumerate(entity_generator):
            e = entities['label'][e_id]
            f = entities['label'][f_id]

            # Training & Test Data
            if i < n_train:
                for source in source_lang:
                    # 90% of facts to learn the rule
                    train.append(e + ' ' + relation[source] + ' ' + f)
                    train.append(f + ' ' + relation[source] + ' ' + e)
            else:
                # 10% we add only one and the other one to test
                if not use_target:
                    for source in source_lang:
                        train.append(e + ' ' + relation[source] + ' ' + f)
                else:
                    for target in target_lang:
                        # Alternatively I never show symmetry in target language but remove the need to transfer
                        train.append(e + ' ' + relation[target] + ' ' + f)

                for target in target_lang:
                    test[target][relation[target]].append(f + ' ' + relation[target] + ' ' + e)

    return train, test, relations_sampled


# Generates asymmetric training data
def generate_anti(relations_symmetric, source_lang, target_lang, n_relations, n_facts):
    train = []
    test = defaultdict(dict)

    # Loading..
    entities, relations = load_data(Relation.Random, source_lang, target_lang, False)

    # Sample relations that aren't symmetric
    idx = list(relations_symmetric['id'])
    relations_filtered = relations.drop(relations[relations.id.isin(idx)].index)
    relations_sampled = relations_filtered.sample(n_relations)

    # For generating entity triples
    entity_triples = []
    old_pairs = []

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for idx, relation in relations_sampled.iterrows():
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            logger.info("ANTI RELATION: " + relation[source] + ' - Frequency: ' + str_count)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][relation[target]] = []

        # Generate n_facts entity2s
        if entity_triples is not None:
            old_pairs = [(triple[1], triple[2]) for triple in entity_triples]
        entity_triples = generate_index_triples(entities.shape[0], entities1, n_facts, old_pairs=old_pairs)

        for i, (e_id, f_id, g_id) in enumerate(entity_triples):
            # Append facts in source lang to training set and target lang to test set.
            e = entities['label'][e_id]
            f = entities['label'][f_id]
            g = entities['label'][g_id]

            # Training Data.
            for source in source_lang:
                # 90% of facts to learn the rule.
                train.append(e + ' ' + relation[source] + ' ' + f)
                train.append(f + ' ' + relation[source] + ' ' + g)

            # Test Data - Testing if ANTI facts are tested as symmetric.
            for target in target_lang:
                test[target][relation[target]].append(f + ' ' + relation[target] + ' ' + e)
                test[target][relation[target]].append(g + ' ' + relation[target] + ' ' + f)

    return train, test, relations_sampled


# Equialence 1 shows that already trained inversions share knowledge among them, even across languages w/ new entities.
def generate_inversion_pretrained(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target):
    train = []
    test = defaultdict(lambda: dict())

    # Split
    n_train = int(0.9 * n_facts)
    n_test_half = int(0.05 * n_facts)

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for index, relation in relations_sampled.iterrows():
        for source in source_lang:
            str_count1 = str(relation['count1']) if 'count1' in relation else 'Unknown'
            str_count2 = str(relation['count2']) if 'count2' in relation else 'Unknown'
            logger.info("RELATION1: " + relation[source + str(1)] + "\t" + "RELATION2: " + relation[source + str(2)])
            logger.info("Relation1 Count: " + str_count1 + "\t" + "Relation2 Count: " + str_count2)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][relation[target + str(1)]] = []
            test[target][relation[target + str(2)]] = []

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for i, (e_id, f_id) in enumerate(entity_generator):
            # Append facts in source lang to training set and target lang to test set.
            e = entities['label'][e_id]
            f = entities['label'][f_id]

            if i < n_train:
                for source in source_lang:
                    # 90% we add both
                    train.append(e + ' ' + relation[source + str(1)] + ' ' + f)
                    train.append(f + ' ' + relation[source + str(2)] + ' ' + e)
            else:
                # 10% we add only one to train and the other one to test
                if i < n_train + n_test_half:
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation[source + str(1)] + ' ' + f)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation[target + str(1)] + ' ' + f)

                    for target in target_lang:
                        test[target][relation[target + str(2)]].append(f + ' ' + relation[target + str(2)] + ' ' + e)
                else:
                    if not use_target:
                        for source in source_lang:
                            train.append(f + ' ' + relation[source + str(2)] + ' ' + e)
                    else:
                        for target in target_lang:
                            train.append(f + ' ' + relation[target + str(2)] + ' ' + e)

                    for target in target_lang:
                        test[target][relation[target + str(1)]].append(e + ' ' + relation[target + str(1)] + ' ' + f)

    return train, test, relations_sampled


# Equivalence 2 shows random relations can become associated as inversions with each other even across languages.
def generate_inversion(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target):
    train = []
    test = defaultdict(dict)

    # Split
    n_train = int(0.9 * n_facts)
    n_test_half = int(0.05 * n_facts)

    # Sample pairs of relations - One relation is the default, the other the inversion
    relations_sampled = relations.sample(2 * n_relations)
    relation_pairs = zip(relations_sampled[::2].iterrows(), relations_sampled[1::2].iterrows())

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for (idx1, relation1), (idx2, relation2) in zip(relations_sampled[::2].iterrows(),
                                                    relations_sampled[1::2].iterrows()):
        for source in source_lang:
            str_count1 = str(relation1['count']) if 'count' in relation1 else 'Unknown'
            str_count2 = str(relation2['count']) if 'count' in relation2 else 'Unknown'
            logger.info("RELATION1: " + relation1[source] + "\t" + "RELATION2: " + relation2[source])
            logger.info("Relation1 Count: " + str_count1 + "\t" + "Relation2 Count: " + str_count2)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][relation1[target]] = []
            test[target][relation2[target]] = []

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for i, (e_id, f_id) in enumerate(entity_generator):
            # Append facts in source lang to training set and target lang to test set.
            e = entities['label'][e_id]
            f = entities['label'][f_id]

            if i < n_train:
                for source in source_lang:
                    # 90% we add both to learn the rule
                    train.append(e + ' ' + relation1[source] + ' ' + f)
                    train.append(f + ' ' + relation2[source] + ' ' + e)
            else:
                # 10% we add only one and the other one to test
                if i < n_train + n_test_half:
                    # 5%
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation1[source] + ' ' + f)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation1[target] + ' ' + f)

                    for target in target_lang:
                        test[target][relation2[target]].append(f + ' ' + relation2[target] + ' ' + e)
                else:
                    # 5%
                    if not use_target:
                        for source in source_lang:
                            train.append(f + ' ' + relation2[source] + ' ' + e)
                    else:
                        for target in target_lang:
                            train.append(f + ' ' + relation2[target] + ' ' + e)

                    for target in target_lang:
                        test[target][relation1[target]].append(e + ' ' + relation1[target] + ' ' + f)

    return train, test, relation_pairs


# e, f, antonym(f)
# Entities are generated in triples where e is consistent across relations (so guessing is hard)
# Generate n_train antonyms per relation, that are then tested on
def generate_negation(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target, n_pairs=100):
    # (e, r, f ) <=> (e, not, antonym(f))
    train = []
    test = defaultdict(lambda: dict())

    # Split
    n_train = int(0.9 * n_facts)
    n_test_half = int(0.05 * n_facts)

    if n_pairs > n_train:
        raise ValueError('n_pair needs to be <= 90% of n_facts.')

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # For generating triples:
    antonym_pairs = []

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for index, relation in relations_sampled.iterrows():
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            logger.info(f"RELATION: {relation[source]} NEGATION: {relation[source + '_negated']}")
            logger.info("Relation - Count: " + str_count)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][relation[target]] = []
            test[target][relation[target + '_negated']] = []

        # Generate n_facts entity2s
        antonym_pairs = generate_negation_triples(entities.shape[0], entities1, n_pairs, old_pairs=antonym_pairs)

        for i, e_id in enumerate(entities1):
            # Append facts in source lang to training set and target lang to test set.
            e = entities['label'][e_id]

            if i < n_train:
                # For Training we have n_pair antonym pairs, if n_pair < n_train, then we cycle through them
                f_id, f_antonym_id = antonym_pairs[i % n_pairs]

                f = entities['label'][f_id]
                f_antonym = entities['label'][f_antonym_id]

                # 90% we add both
                for source in source_lang:
                    train.append(e + ' ' + relation[source] + ' ' + f)
                    train.append(e + ' ' + relation[source + '_negated'] + ' ' + f_antonym)
            else:
                # After training we just sample random antonym pairs that were seen during training
                f_id, f_antonym_id = sample(antonym_pairs, 1)[0]

                f = entities['label'][f_id]
                f_antonym = entities['label'][f_antonym_id]

                # 10% we add only one to train and the other one to test
                if i < n_train + n_test_half:
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation[source] + ' ' + f)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation[target] + ' ' + f)

                    for target in target_lang:
                        test[target][relation[target + '_negated']].append(
                            e + ' ' + relation[target + '_negated'] + ' ' + f_antonym)
                else:
                    if not use_target:
                        for source in source_lang:
                            train.append(e + ' ' + relation[source + '_negated'] + ' ' + f_antonym)
                    else:
                        for target in target_lang:
                            train.append(e + ' ' + relation[target + '_negated'] + ' ' + f_antonym)

                    for target in target_lang:
                        test[target][relation[target]].append(e + ' ' + relation[target] + ' ' + f)

    return train, test, relations_sampled


# (e, r, f) => (e, s, a1), (e, s, a2), ...
def generate_implication(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target, precision_k,
                         n_pairs, use_enhanced):
    train = []
    test = defaultdict(dict)

    # Split
    n_train = int(0.9 * n_facts)

    if n_pairs > n_train:
        raise ValueError('n_pair needs to be below 90% of n_facts.')

    # Sample pairs of relations - Relation and its implicated relation
    relations_sampled = relations.sample(2 * n_relations)
    relation_pairs = zip(relations_sampled[::2].iterrows(), relations_sampled[1::2].iterrows())

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for (idx1, relation), (idx2, implication) in zip(relations_sampled[::2].iterrows(),
                                                     relations_sampled[1::2].iterrows()):
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            str_count_imp = str(implication['count']) if 'count' in implication else 'Unknown'
            logger.info("RELATION: " + relation[source] + "\t" + "IMPLICATION: " + implication[source])
            logger.info("Relation Count: " + str_count + "\t" + "Implication Count: " + str_count_imp)

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][implication[target]] = []

        # Generate n_pairs entities that are then (re-)used
        implied_indices = generate_index_implication(entities.shape[0], entities1, n_pairs, precision_k)

        if use_enhanced:
            for idx in implied_indices:
                f = entities['label'][idx[0]]

                for k in range(precision_k):
                    a = entities['label'][idx[1 + k]]
                    train.append(f + ' connected ' + a)

        # Append facts in source lang to training set and target lang to test set.
        for i, e_id in enumerate(entities1):
            # Get entity labels
            e = entities['label'][e_id]

            if i < n_train:
                # Implied training facts are at least seen once during training
                f_id = implied_indices[i % n_pairs][0]
                f = entities['label'][f_id]

                entities_implied = []
                for k in range(precision_k):
                    entities_implied.append(entities['label'][implied_indices[i % n_pairs][1 + k]])

                for source in source_lang:
                    # 90% of facts learn the rule
                    train.append(e + ' ' + relation[source] + ' ' + f)

                    for a in entities_implied:
                        train.append(e + ' ' + implication[source] + ' ' + a)
            else:
                # For testing we sample some index tuple
                implication_idx = sample(implied_indices, 1)[0]

                f = entities['label'][implication_idx[0]]

                entities_implied = []
                for k in range(precision_k):
                    entities_implied.append(entities['label'][implication_idx[1 + k]])

                # 10% for testing
                if not use_target:
                    for source in source_lang:
                        train.append(e + ' ' + relation[source] + ' ' + f)
                else:
                    for target in target_lang:
                        train.append(e + ' ' + relation[target] + ' ' + f)

                for target in target_lang:
                    for a in entities_implied:
                        test[target][implication[target]].append(e + ' ' + implication[target] + ' ' + a)

    return train, test, relation_pairs


# (e, r, f) and (f, s, g) => (e, t, g)
def generate_composition(entities, relations, source_lang, target_lang, n_relations, n_facts, n_pairs, use_target, use_same_relations):
    train = []
    test = defaultdict(dict)

    # Split
    n_train = int(0.9 * n_facts)

    if use_same_relations:
        relations_sampled = relations.sample(n_relations)
        relation_triples = zip(relations_sampled.iterrows(), relations_sampled.iterrows(), relations_sampled.iterrows())
        relations_iter = zip(relations_sampled.iterrows(), relations_sampled.iterrows(), relations_sampled.iterrows())
    else:
        # Sample pairs of relations - Relation and its implicated relation
        relations_sampled = relations.sample(3 * n_relations)
        relation_triples = zip(relations_sampled[::3].iterrows(),
                               relations_sampled[1::3].iterrows(),
                               relations_sampled[2::3].iterrows())
        relations_iter = zip(relations_sampled[::3].iterrows(),
                             relations_sampled[1::3].iterrows(),
                             relations_sampled[2::3].iterrows())

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for (idx1, relation1), (idx2, relation2), (idx3, composition) in relations_iter:
        for source in source_lang:
            str_count1 = str(relation1['count']) if 'count' in relation1 else 'Unknown'
            str_count2 = str(relation2['count']) if 'count' in relation2 else 'Unknown'
            str_count_comp = str(composition['count']) if 'count' in composition else 'Unknown'
            logger.info(f"RELATION1: {relation1[source]} and RELATION2: {relation2[source]} => COMPOSITION: {composition[source]}")
            logger.info(f"Relation1 Count: {str_count1}, Relation2 Count: {str_count2}, Composition Count: {str_count_comp}")

        for target in target_lang:
            test[target][composition[target]] = []

        # Generate f and g entities and don't repeat them for the next relation
        composition_indices = generate_index_composition(entities.shape[0], entities1, n_pairs)

        for i, e_id in enumerate(entities1):
            # Append facts in source lang to training set and target lang to test set.
            e = entities['label'][e_id]

            f_id = composition_indices[i % n_pairs][0]
            g_id = composition_indices[i % n_pairs][1]
            f = entities['label'][f_id]
            g = entities['label'][g_id]

            if i < n_train:
                for source in source_lang:
                    # 90% of facts learn the rule
                    train.append(e + ' ' + relation1[source] + ' ' + f)
                    train.append(f + ' ' + relation2[source] + ' ' + g)
                    # =>
                    train.append(e + ' ' + composition[source] + ' ' + g)
            else:
                # 10% for testing
                if not use_target:
                    for source in source_lang:
                        train.append(e + ' ' + relation1[source] + ' ' + f)
                        train.append(f + ' ' + relation2[source] + ' ' + g)
                else:
                    for target in target_lang:
                        train.append(e + ' ' + relation1[target] + ' ' + f)
                        train.append(f + ' ' + relation2[target] + ' ' + g)

                for target in target_lang:
                    test[target][composition[target]].append(e + ' ' + composition[target] + ' ' + g)

    return train, test, relation_triples


# Based more on symbolic reasoner (NOT WORKING)
# Each entities inside a group are considered equal
def generate_composition_enhanced(entities, relations, source_lang, target_lang, n_relations, n_facts, use_target, use_same_relations):
    train = []
    test = defaultdict(dict)

    # Sample pairs of relations - Relation and its implicated relation
    if use_same_relations:
        relations_sampled = relations.sample(n_relations)
        relation_triples = zip(relations_sampled.iterrows(), relations_sampled.iterrows(), relations_sampled.iterrows())
        relations_iter = zip(relations_sampled.iterrows(), relations_sampled.iterrows(), relations_sampled.iterrows())
    else:
        # Sample pairs of relations - Relation and its implicated relation
        relations_sampled = relations.sample(3 * n_relations)
        relation_triples = zip(relations_sampled[::3].iterrows(),
                               relations_sampled[1::3].iterrows(),
                               relations_sampled[2::3].iterrows())
        relations_iter = zip(relations_sampled[::3].iterrows(),
                             relations_sampled[1::3].iterrows(),
                             relations_sampled[2::3].iterrows())

    for (idx1, relation1), (idx2, relation2), (idx3, composition) in relations_iter:
        for source in source_lang:
            str_count1 = str(relation1['count']) if 'count' in relation1 else 'Unknown'
            str_count2 = str(relation2['count']) if 'count' in relation2 else 'Unknown'
            str_count_comp = str(composition['count']) if 'count' in composition else 'Unknown'
            logger.info(f"RELATION1: {relation1[source]} and RELATION2: {relation2[source]} => COMPOSITION: {composition[source]}")
            logger.info(f"Relation1 Count: {str_count1}, Relation2 Count: {str_count2}, Composition Count: {str_count_comp}")

        for target in target_lang:
            # Create Test {'target':{'relation_target':['e1 r e2', 'e3 r e4', ...]}
            test[target][composition[target]] = []

        # Copy entities to not modify the original
        entities_copy = entities.copy()

        # Multiple Groupings per relation
        for _ in range(5):
            # Holds the composition that we need to split for testing
            train_composition = []

            # Sample 30 Entities that are then grouped in E, F, G
            entities1 = entities_copy['label'].sample(30)

            # Remove entities to get uniqueness within this relation
            entities_copy = entities_copy.drop(list(entities1.index))

            # Create entity groups
            e_group = entities1[:10]
            f_group = entities1[10:20]
            g_group = entities1[20:]

            # Create group associations
            for group in [e_group, f_group, g_group]:
                # Creates all pairs without repetition n*(n-1)
                for e1, e2 in itertools.permutations(group, 2):
                    train.append(e1 + ' connected ' + e2)  # 'connected' is a single token

            # TRAIN
            for source in source_lang:
                for e in e_group:
                    for f in f_group:
                        train.append(e + ' ' + relation1[source] + ' ' + f)
                # and
                for f in f_group:
                    for g in g_group:
                        train.append(f + ' ' + relation2[source] + ' ' + g)
                # =>
                for e in e_group:
                    for g in g_group:
                        train_composition.append((e, g))

            # Take 90% for compositions training
            for (e, g) in train_composition[:90]:
                for source in source_lang:
                    train.append(e + ' ' + composition[source] + ' ' + g)

            # Rest for testing 10%
            for (e, g) in train_composition[90:]:
                for target in target_lang:
                    test[target][composition[target]].append(e + ' ' + composition[target] + ' ' + g)

    return train, test, relation_triples


# Used to generate random facts for the entities
def generate_random(source_lang, target_lang, n_facts, n_relations):
    train = []

    entities, relations = load_data(Relation.Random, source_lang, target_lang, False)

    # Sample relations
    relations_sampled = relations.sample(n_relations)

    # Sample n_facts entity1s, which we repeat for every relation but with different entity2
    entities1 = generate_unique_indices(entities.shape[0], n_facts)

    for idx, relation in relations_sampled.iterrows():
        for source in source_lang:
            str_count = str(relation['count']) if 'count' in relation else 'Unknown'
            logger.info("RANDOM RELATION: " + relation[source] + ' - Count: ' + str_count)

        # Generate n_facts entity2s
        entity_generator = generate_index_pairs(entities.shape[0], entities1, n_facts)

        for i, (e_id, f_id) in enumerate(entity_generator):
            e = entities['label'][e_id]
            f = entities['label'][f_id]

            # Training & Test Data
            for source in source_lang:
                # 90% of facts to learn the rule
                train.append(e + ' ' + relation[source] + ' ' + f)

    return train, relations_sampled
