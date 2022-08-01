import argparse
import copy

from transformers import BertForMaskedLM, BertTokenizerFast, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, IntervalStrategy
from datasets import Dataset
import os

from data_generation_relation import *
from utils import *
from custom_trainer import CustomTrainer
from datasets import load_metric
import logging
from transformers import logging as tlogging
import wandb
import sys
from utils import set_seed
from transformers.integrations import WandbCallback, TensorBoardCallback

class Relation(Enum):
    Equivalence = 'equivalence'
    Symmetry = 'symmetry'
    Inversion = 'inversion'
    Negation = 'negation'
    Implication = 'implication'
    Composition = 'composition'
    Random = 'random'

    def __str__(self):
        return self.value


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


def evaluation_equivalence_pretrained(trainer, tokenizer, relations, source_language, test):
    for target in test:

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            for source in source_language:
                print(f'Relation - source: {relation[source]}, target: {relation[target]}')
                print(f'Alias - source: {relation[source + "_alias"]}, target: {relation[target + "_alias"]}')

            # RELATION
            if test[target][relation[target]]:
                # Relation from test set dict
                relation_test = test[target][relation[target]]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)

            # ALIAS
            if test[target][relation[target + '_alias']]:

                # Relation from test set dict
                relation_test = test[target][relation[target + '_alias']]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)
            
            print('')
            print('')


def evaluation_equivalence(trainer, tokenizer, relation_pairs, source_language, test):
    for target in test:

        # Iterate over all relations per target language
        for (idx1, relation1), (idx2, relation2) in zip(relation_pairs[0].iterrows(), relation_pairs[1].iterrows()):
            for source in source_language:
                print(f'Relation1 - source: {relation1[source]}, target: {relation1[target]}')
                print(f'Relation2 - source: {relation2[source]}, target: {relation2[target]}')
            
            # RELATION1
            if test[target][relation1[target]]:
                # Relation from test set dict
                relation_test = test[target][relation1[target]]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)

            # RELATION2
            if test[target][relation2[target]]:
                # Relation from test set dict
                relation_test = test[target][relation2[target]]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)


def evaluation_symmetry(trainer, tokenizer, relations, source_language, test):
    for target in test:

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            for source in source_language:
                print(f'Relation - source: {relation[source]}, target: {relation[target]}')

            if not test[target][relation[target]]:
                continue

            # Relation from test set dict
            relation_test = test[target][relation[target]]

            # Tokenize
            relation_test_ds = Dataset.from_dict({'sample': relation_test})
            tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

            # Evaluate
            metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
            output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
            print(output_metrics)


def evaluation_inversion_pretrained(trainer, tokenizer, relations, source_language, test):
    for target in test:

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            # RELATION
            for source in source_language:
                print(f"Relation - source: {relation[source + str(1)]} Relation - target: {relation[target + str(1)]}")

            if not test[target][relation[target + str(1)]]:
                continue

            # Relation from test set dict
            relation_test = test[target][relation[target + str(1)]]

            # Tokenize
            relation_test_ds = Dataset.from_dict({'sample': relation_test})
            tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

            # Evaluate
            metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
            output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
            print(output_metrics)

            # INVERSION
            for source in source_language:
                logger.info(
                    f"Inversion - source: {relation[source + str(2)]} Alias - target: {relation[target + str(2)]}")

            if not test[target][relation[target + str(2)]]:
                continue

            # Relation from test set dict
            relation_test = test[target][relation[target + str(2)]]

            # Tokenize
            relation_test_ds = Dataset.from_dict({'sample': relation_test})
            tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

            # Evaluate
            metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
            output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
            print(output_metrics)


def evaluation_inversion(trainer, tokenizer, relation_pairs, source_language, test):
    for target in test:

        # Iterate over all relations per target language
        for (idx1, relation1), (idx2, relation2) in zip(relation_pairs[0].iterrows(), relation_pairs[1].iterrows()):

            # RELATION
            for source in source_language:
                print(f'Relation - source: {relation1[source]}, target: {relation1[target]}')

            if test[target][relation1[target]]:

                # Relation from test set dict
                relation_test = test[target][relation1[target]]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)

                # INVERSION
                for source in source_language:
                    print(f'Inversion - source: {relation2[source]}, target: {relation2[target]}')

            if test[target][relation2[target]]:
                # Relation from test set dict
                relation_test = test[target][relation2[target]]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)


def evaluation_negation(trainer, tokenizer, relations, source_language, test):
    for target in test:

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            # RELATION
            for source in source_language:
                print(f"Relation - source: {relation[source]} Relation - target: {relation[target]}")

            if test[target][relation[target]]:
                # Relation from test set dict
                relation_test = test[target][relation[target]]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)

            # NEGATION
            for source in source_language:
                print(f"Negation - source: {relation[source + '_negated']} Alias - target: {relation[target + '_negated']}")
                
            if test[target][relation[target + '_negated']]:
                # Relation from test set dict
                relation_test = test[target][relation[target + '_negated']]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)

            
def evaluation_implication(trainer, tokenizer, relation_pairs, test):
    for target in test:

        # Iterate over all relations per target language
        for (idx1, relation), (idx2, implication) in zip(relation_pairs[0].iterrows(), relation_pairs[1].iterrows()):
            # IMPLICATION
            if not test[target][implication[target]]:
                continue

            # Relation from test set dict
            relation_test = test[target][implication[target]]

            # Tokenize
            relation_test_ds = Dataset.from_dict({'sample': relation_test})
            tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

            # Evaluate
            metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
            output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
            print(output_metrics)


def evaluation_composition(trainer, tokenizer, relation_triples, source_language, test):
    for target in test:

        # Iterate over all relations per target language
        for (idx1, relation1), (idx2, relation2), (idx3, composition) in zip(relation_triples[0].iterrows(),
                                                                             relation_triples[1].iterrows(),
                                                                             relation_triples[2].iterrows()):
            for source in source_language:
                print(f"Relation - source: {relation1[source]} Relation - target: {relation1[target]}")
                print(f"Relation - source: {relation2[source]} Relation - target: {relation2[target]}")

                print(f"Relation - source: {composition[source]} Relation - target: {composition[target]}")
                
            
            # IMPLICATION
            if not test[target][composition[target]]:
                continue

            # Relation from test set dict
            relation_test = test[target][composition[target]]

            # Tokenize
            relation_test_ds = Dataset.from_dict({'sample': relation_test})
            tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

            # Evaluate
            metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
            output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
            print(output_metrics)

