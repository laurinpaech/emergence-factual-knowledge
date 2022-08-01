import argparse
from transformers import TrainingArguments, DataCollatorForLanguageModeling, IntervalStrategy, AutoTokenizer, \
    AutoModelForMaskedLM
from datasets import Dataset
import os

from transformers.integrations import WandbCallback, TensorBoardCallback

from utils import *
from custom_trainer import CustomTrainer
from data_generation_knowledge import *
from datasets import load_metric
import pandas as pd
import copy
from transformers import logging as tlogging
import wandb
import sys


def tokenize_dict_keys(input_dict, tokenizer):
    test_dict = copy.deepcopy(input_dict)

    for old_key in input_dict:
        # Tokenize old keys + Remove last token of each key
        key_token = tokenizer(old_key)['input_ids'][:-1]

        # Replace string old_key with tokenized key, value is old value
        test_dict[repr(key_token)] = test_dict.pop(old_key)

    return test_dict


# Mean reciprocal rank
def mean_reciprocal_rank(eval_pred):
    relation_logits, relation_labels = eval_pred

    indices = np.where(relation_labels != -100)  # Select only the ones that are masked
    relation_preds = relation_logits[indices]
    relation_true_labels = relation_labels[indices]

    # Compute ranks and sum inverses.
    inv_rank_sum = 0

    # For every sample
    n = relation_preds.shape[0]
    for i in range(n):
        # Find the rank
        rank_idx = np.where(relation_preds[i] == relation_true_labels[i])[0]
        if rank_idx.size == 0:
            inv_rank_sum += 1 / 1000
        else:
            inv_rank_sum += 1 / rank_idx[0]
    # Average
    mrr = inv_rank_sum / n

    # Compute correct predictions
    correct_predictions = []
    for i, prediction in enumerate(relation_preds):
        # Take top k of prediction and see if label is in it
        if relation_true_labels[i] in prediction:
            correct_predictions.append(True)
        else:
            correct_predictions.append(False)

    return {'eval_accuracy': mrr, 'correct_predictions': correct_predictions}


# Metric for Precision@1
def precision_at_one(eval_pred):
    metric = load_metric("accuracy")
    relation_logits, relation_labels = eval_pred

    # Relation Accuracy
    indices = np.where(relation_labels != -100)  # Select only the ones that are masked
    correct_predictions = relation_logits[indices] == relation_labels[indices]
    relation_precision = metric.compute(predictions=relation_logits[indices],
                                        references=relation_labels[indices])['accuracy']
    return {'eval_accuracy': relation_precision, 'correct_predictions': correct_predictions}


# Metric for Precision@1
def precision_at_one_alias(eval_pred, alias_pred):
    relation_logits, relation_labels = eval_pred
    alias_logits, alias_labels = alias_pred

    # Relation Accuracy
    indices = np.where(relation_labels != -100)  # Select only the ones that are masked
    relation_preds = relation_logits[indices]
    relation_true_labels = relation_labels[indices]

    accumulator = 0
    correct_predictions = []
    # Iterate over all facts
    for i, alias_pred_list in enumerate(alias_logits):
        # Relation fact
        pred = relation_preds[i]
        true_label = relation_true_labels[i]

        if pred == true_label:
            accumulator += 1
            correct_predictions.append(True)
            continue

        # Alias facts
        indices_alias = np.where(alias_labels[i] != -100)
        for j, alias_pred in enumerate(alias_pred_list):
            # See if alias has predicted correctly
            alias_pred_i = alias_pred[indices_alias[1][j]].item()
            # if alias_pred_i == true_label:
            if alias_pred_i == true_label:
                accumulator += 1
                correct_predictions.append(True)
                break

        # for..else structure means else is executed if there was no break
        else:
            correct_predictions.append(False)

    relation_precision = accumulator / len(relation_preds)
    return {'eval_accuracy': relation_precision, 'correct_predictions': correct_predictions}


# Metric for Precision@K.
def precision_at_k(eval_pred, k_list):
    relation_logits, relation_labels = eval_pred

    # Relation Accuracy
    indices = np.where(relation_labels != -100)  # Select only the ones that are masked
    relation_preds = relation_logits[indices]
    relation_true_labels = relation_labels[indices]

    accumulator = 0
    correct_predictions = []
    for i, prediction in enumerate(relation_preds):
        # Top k?
        k = k_list[i]
        # Take top k of prediction and see if label is in it
        if relation_true_labels[i] in prediction[:k]:
            accumulator += 1
            correct_predictions.append(True)
        else:
            correct_predictions.append(False)

    relation_precision = accumulator / len(relation_preds)
    return {'eval_accuracy': relation_precision, 'correct_predictions': correct_predictions}


# Metric for Precision@K but with fixed k.
def precision_at_k_fixed(eval_pred):
    relation_logits, relation_labels = eval_pred

    # Relation Accuracy
    indices = np.where(relation_labels != -100)  # Select only the ones that are masked
    relation_preds = relation_logits[indices]
    relation_true_labels = relation_labels[indices]

    accumulator = 0
    correct_predictions = []
    for i, prediction in enumerate(relation_preds):
        # Take top k of prediction and see if label is in it
        if relation_true_labels[i] in prediction:
            accumulator += 1
            correct_predictions.append(True)
        else:
            correct_predictions.append(False)

    relation_precision = accumulator / len(relation_preds)
    return {'eval_accuracy': relation_precision, 'correct_predictions': correct_predictions}


# ~~ TOKENIZE ~~
def tokenize(tokenizer, dataset):
    def tokenize_fn(examples):
        result = tokenizer(examples["sample"])
        return result

    # Use batched=True to activate fast multithreading!
    tokenized_ds = dataset.map(
        tokenize_fn, batched=True, remove_columns=["sample"]
    )

    return tokenized_ds


def evaluation_alias(trainer, tokenizer, relations, source_language, target_language, test, test_key='relation'):
    # Create separate language Datasets for evaluation
    test_datasets = []

    # Update the precision dict for alias with test_key (alias, translation
    precision_k = defaultdict(int)
    for lang in test:
        for relation in test[lang]:
            for alias_relation in test[lang][relation][test_key]:
                for subj_rel in test[lang][relation][test_key][alias_relation]:
                    precision_k[subj_rel] = len(test[lang][relation][test_key][alias_relation][subj_rel])
    trainer.precision_dict = tokenize_dict_keys(precision_k, tokenizer)

    # Reformat test
    test = test_to_normal_alias(test, test_key)
    test_relations = flatten_remove_dict4(test, test_key)

    for language_code in test.keys():
        language_ds = Dataset.from_dict({language_code: test_relations[language_code]})
        test_datasets.append(language_ds)

    # Iterate over relations to evaluate
    for _, relation in relations.iterrows():
        for source in source_language:
            print(f'RELATION: {relation[source]}')

        # Iterate over all relations per target language
        for test_lang in target_language:
            print(f'-> LANGUAGE: {test_lang} - TARGET: {relation[test_lang]}')

            if test_key == 'alias':
                logger.info("Number of Aliases: %i", len(test_relations[test_lang][relation[test_lang]]))
            elif test_key == 'translate':
                logger.info("Number of Translations: %i", len(test_relations[test_lang][relation[test_lang]]))
            elif test_key == 'subword':
                logger.info("Number of Words: %i", len(test_relations[test_lang][relation[test_lang]]))

            # Skip if no relations to evaluate
            if not test_relations[test_lang][relation[test_lang]]:
                continue

            for alias in test_relations[test_lang][relation[test_lang]]:
                # Relation from test set dict
                relation_test = test_relations[test_lang][relation[test_lang]][alias]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                logger.info(test_key.title() + ': ' + alias)
                print(trainer.evaluate(eval_dataset=tokenized_relation_ds))


def evaluation_relation(trainer, tokenizer, relations, source_language, target_language, run_name, test):
    # Create separate language Datasets for evaluation
    test_datasets = []

    # Test needs to have all facts as list but
    test = test_to_normal(test)
    test_relations = flatten_remove_dict(test, 'relation')

    for language_code in test.keys():
        language_ds = Dataset.from_dict({language_code: test_relations[language_code]})
        test_datasets.append(language_ds)

    # Dict of dataframe per relation in every language with all entities, key is in source lang
    entity_acc_per_relation = {relation[source]: pd.DataFrame() for source in source_language for _, relation in
                               relations.iterrows()}
    entity_acc_per_language = {language_code: pd.DataFrame() for language_code in test}

    # Iterate over target languages to evaluate
    for test_lang in target_language:
        # Dictionary {entityA: {relationA: correct/total, relationB: correct/total}, entityB: {...}}
        entity_relation_acc = {}

        # Iterate over all relations per target language
        for _, relation in relations.iterrows():
            for source in source_language:
                print(f'Relation - source: {relation[source]}, target: {relation[test_lang]}')

            if not test_relations[test_lang][relation[test_lang]]:
                continue

            # Relation from test set dict
            relation_test = test_relations[test_lang][relation[test_lang]]

            # Tokenize
            relation_test_ds = Dataset.from_dict({'sample': relation_test})
            tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

            # Evaluate
            metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
            output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
            print(output_metrics)

            # Compute number of occurences for this relation
            entity_occurences = compute_occurences(relation_test, metrics, False)

            # *** DATAFRAME PER LANGUAGE FOR EVERY RELATION WITH ALL ENTITIES ***
            for entity in entity_occurences:
                if entity not in entity_relation_acc:
                    entity_relation_acc[entity] = {}

                # Compute the correctly predited ratio for every entity appearing in this relation
                correct = entity_occurences[entity]['correct_subject'] + entity_occurences[entity]['correct_object']
                total = entity_occurences[entity]['subject'] + entity_occurences[entity]['object']

                for source in source_language:
                    entity_relation_acc[entity][relation[source]] = correct / total

            # *** DATAFRAME PER RELATION IN EVERY LANGUAGE WITH ALL ENTITIES ***
            # If there is no column entities? Create it
            for source in source_language:
                df = entity_acc_per_relation[relation[source]]
                if 'entity' not in df or len(df['entity']) < len(entity_occurences.keys()):
                    # If there is a mismatch in entity number (due to taking some for validation), reindex and inc df
                    df = df.reindex(range(0, len(entity_occurences.keys())))
                    df['entity'] = entity_occurences.keys()

                # Compute entity accuracy
                acc_subj = []
                acc_obj = []
                total_subj = []
                total_obj = []
                for entity in entity_occurences:
                    entity_dict = entity_occurences[entity]
                    if entity_dict['subject'] == 0:
                        acc_subj.append(0)
                        total_subj.append(0)
                    else:
                        acc_subj.append(entity_dict['correct_subject'] / entity_dict['subject'])
                        total_subj.append(entity_dict['subject'])

                    if entity_dict['object'] == 0:
                        acc_obj.append(0)
                        total_obj.append(0)
                    else:
                        acc_obj.append(entity_dict['correct_object'] / entity_dict['object'])
                        total_obj.append(entity_dict['object'])

                # Add results for this language relation to its relation-dataframe
                df[test_lang + '_subject'] = acc_subj
                df[test_lang + '_subject_total'] = total_subj
                df[test_lang + '_object'] = acc_obj
                df[test_lang + '_object_total'] = total_obj

                # Since reindex is a copy, we need to copy it back
                entity_acc_per_relation[relation[source]] = df
            # ***

        # Entity/Relation Accuracy Dict to Dataframe (and Transpose so that entities are on rows)
        entity_acc_per_language[test_lang] = pd.DataFrame(entity_relation_acc).T

    # Save entity results
    results_dir = './output/' + run_name + '/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for relation in entity_acc_per_relation:
        entity_acc_per_relation[relation].to_csv('./output/' + run_name + '/results/' + relation + '.csv',
                                                 index=False)

    for language in entity_acc_per_language:
        entity_acc_per_language[language].to_csv('./output/' + run_name + '/results/' + language + '.csv',
                                                 index=False)


def compute_occurences(relation_test, metrics, target_sov):
    # TODO: This could be done way better probably during data generation
    entity_occurences = {}
    for i, fact in enumerate(relation_test):
        entity1 = fact.split(' ')[0]
        if target_sov:
            entity2 = fact.split(' ')[1]
        else:
            entity2 = fact.split(' ')[-1]

        # Count entity occurences
        if entity1 in entity_occurences:
            entity_occurences[entity1]['subject'] += 1
        else:
            # Create it
            entity_occurences[entity1] = {}
            entity_occurences[entity1]['subject'] = 1
            entity_occurences[entity1]['object'] = 0
            entity_occurences[entity1]['correct_subject'] = 0
            entity_occurences[entity1]['correct_object'] = 0

        # Count entity occurences
        if entity2 in entity_occurences:
            entity_occurences[entity2]['object'] += 1
        else:
            # Create it
            entity_occurences[entity2] = {}
            entity_occurences[entity2]['object'] = 1
            entity_occurences[entity2]['correct_object'] = 0
            entity_occurences[entity2]['subject'] = 0
            entity_occurences[entity2]['correct_subject'] = 0

        entity_occurences[entity1]['correct_subject'] += int(metrics['eval_correct_predictions'][i])
        entity_occurences[entity2]['correct_object'] += int(metrics['eval_correct_predictions'][i])

    return entity_occurences


if __name__ == "__main__":
    main()
