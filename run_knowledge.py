import argparse
from transformers import TrainingArguments, DataCollatorForLanguageModeling, IntervalStrategy, AutoTokenizer, \
    AutoModelForMaskedLM, BertConfig
from datasets import Dataset
import os

from transformers.integrations import TensorBoardCallback

from util.utils import *
from custom_trainer import CustomTrainer
from data_generation_knowledge import *
from datasets import load_metric
import pandas as pd
import copy
import logging
from transformers import logging as tlogging
import sys

logger = logging.getLogger(__name__)


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
            # inv_rank_sum += 1 / 1000
            inv_rank_sum += 0
        else:
            inv_rank_sum += 1 / (rank_idx[0] + 1)
    # Average
    mrr = inv_rank_sum / n

    if not 0 <= mrr <= 1:
        raise ValueError(f'MRR is out of range! {n} samples, MRR {mrr}')

    # Compute correct predictions
    correct_predictions = []
    for i, prediction in enumerate(relation_preds):
        # Take top k of prediction and see if label is in it
        if relation_true_labels[i] in prediction:
            correct_predictions.append(True)
        else:
            correct_predictions.append(False)

    return {'eval_accuracy': mrr, 'correct_predictions': correct_predictions}


# Mean reciprocal rank
def mean_reciprocal_rank_alias(eval_pred, alias_pred):
    relation_logits, relation_labels = eval_pred
    alias_logits, alias_labels = alias_pred

    indices = np.where(relation_labels != -100)  # Select only the ones that are masked
    relation_preds = relation_logits[indices]
    relation_true_labels = relation_labels[indices]

    # Compute ranks and sum inverses.
    inv_rank_sum = 0
    correct_predictions = []

    # For every sample
    for i, alias_pred_list in enumerate(alias_logits):
        # Relation fact
        prediction = relation_preds[i]
        true_label = relation_true_labels[i]

        # Take top k of prediction and see if label is in it
        if true_label in prediction:
            # Find the rank
            rank_idx = np.where(relation_preds[i] == relation_true_labels[i])[0]

            inv_rank_sum += 1 / (rank_idx[0] + 1)
            correct_predictions.append(True)
            continue

        # Alias facts
        indices_alias = np.where(alias_labels[i] != -100)
        for j, alias_pred in enumerate(alias_pred_list):
            # See if alias has predicted correctly
            alias_pred_i = alias_pred[indices_alias[1][j]]
            if true_label in alias_pred_i:
                # Find the rank
                rank_idx = np.where(relation_preds[i] == relation_true_labels[i])[0]

                inv_rank_sum += 1 / (rank_idx[0] + 1)
                correct_predictions.append(True)
                break

        # for..else structure means else is executed if there was no break
        else:
            correct_predictions.append(False)

    # Average
    mrr = inv_rank_sum / relation_preds.shape[0]

    if not 0 <= mrr <= 1:
        raise ValueError(f'MRR is out of range! {relation_preds.shape[0]} samples, MRR {mrr}')

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

        # Check if prediction is correct, if yes, continue, if not check aliases.
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


def precision_at_k_fixed_alias(eval_pred, alias_pred):
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

        # Check if prediction is correct, if yes, continue, if not check aliases.
        if true_label in pred:
            accumulator += 1
            correct_predictions.append(True)
            continue

        # Alias facts
        indices_alias = np.where(alias_labels[i] != -100)
        for j, alias_pred in enumerate(alias_pred_list):
            # See if alias has predicted correctly
            alias_pred_i = alias_pred[indices_alias[1][j]]
            if true_label in alias_pred_i:
                accumulator += 1
                correct_predictions.append(True)
                break

        # for..else structure means else is executed if there was no break
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


# Investigate knowledge transferability between source and target language
def main():
    # PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '-r', type=str, default="KnowledgeTransferDefault",
                        help='How to name current run')
    parser.add_argument('--source_language', '-s', nargs='+', default=['en'], help='source language')
    parser.add_argument('--target_language', '-t', nargs='+', default=['de', 'es', 'fr'], help='target languages')
    parser.add_argument('--n_relations', type=int, default='10')
    parser.add_argument('--n_facts', type=int, default='1000')
    parser.add_argument('--epochs', type=int, default='200', help='Default is 200 epochs')
    parser.add_argument('--batch_size', type=int, default='256', help='batch size per device')
    parser.add_argument('--lr', type=float, default='6e-5')
    parser.add_argument('--bert', action='store_true', help='Use BERT as model instead of mBERT')
    parser.add_argument('--use_fixed_relations', action='store_true', help='Use pre-selected relations for KT. '
                                                                           'Seed only varies entities now.')
    parser.add_argument("--no_alias", '-a', action='store_true', help="not evaluating aliases, translations, subwords")
    parser.add_argument("--evaluate_test", action='store_true', help="evaluate on test set and skip validation")
    parser.add_argument("--combined_metric", action='store_true',
                        help="evaluate with less selection bias on relations and their aliases")
    parser.add_argument("--train_w_alias", action='store_true', help="training with aliases, translations")
    parser.add_argument("--multilingual", action='store_true', help="use multilingual entities (no agnostic)")
    parser.add_argument("--multilingual_object", action='store_true',
                        help="use multilingual object but subject as agnostic")
    parser.add_argument("--multilingual_subject", action='store_true',
                        help="use multilingual subject but agnostic object")
    parser.add_argument("--target_sov", action='store_true', help="use sov for target")
    parser.add_argument("--source_sov", action='store_true', help="use sov for source")
    parser.add_argument("--source_entities", action='store_true', help="use w multilingual entites, only source lang "
                                                                       "entities")
    parser.add_argument("--n_shot", type=int, default=0, help="Use N-shot learning")
    parser.add_argument("--verify_model", action='store_true', help="verify that the facts are not predicted by model")
    parser.add_argument("--frequency_test", action='store_true')
    parser.add_argument("--reuse_test", action='store_true')
    parser.add_argument("--cs_test", action='store_true')
    parser.add_argument("--dot_test", action='store_true')
    parser.add_argument("--precision_k", type=int, default=0)
    parser.add_argument("--metric_mrr", action='store_true', help='Use mrr metric in training')
    parser.add_argument("--subject_per_relation", type=int, default='1',
                        help='How often can an entity be repeated as Subject-Entity.')
    parser.add_argument("--subject_all_relation", type=int, default='10',
                        help='How often can entity be repeated as Subject-Entity across relations')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--n_gpu", type=int, default=2, help="single or two gpus")
    args = parser.parse_args()
    args.use_alias = not args.no_alias

    # SANITY CHECK
    # Check we are using same script for entities and relations
    languages = args.source_language + args.target_language
    if not args.multilingual and any(e in ['zh', 'ru', 'ja'] for e in languages):
        # This would mix english entities + chinese/russian/japanese relations
        logger.warning('Using latin entities with relations in non-latin script!')

    if args.multilingual_object or args.multilingual_subject:
        if args.frequency_test or args.cs_test:
            logger.warning('Agnostic entities mixed with multilingual dont work for frequency or cs test!')
        args.multilingual = True

    # Check that source and target languages don't overlap
    if list(set(args.source_language) & set(args.target_language)):
        logger.warning('Overlapping source and target languages!')

    # In general subjects will be reused as much as possible, so this is just a limit!
    # Range: 1 - n_relations
    if args.reuse_test and (args.subject_all_relation > args.n_relations or args.subject_all_relation < 1):
        args.subject_all_relation = args.n_relations

    if args.reuse_test:
        logger.info('Running Experiments on Re-Use of Subjects!')

    if args.frequency_test:
        logger.info('Running Experiments on Frequency of facts!')

    if args.cs_test:
        logger.info('Running Codeswitching Experiment!')

    # Fix randomness
    set_seed(args.seed)

    # GPU Settings
    if args.n_gpu == 1:
        # Use only single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif args.n_gpu == 0:
        # Use CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        args.n_gpu = torch.cuda.device_count()

    # LOGGING Setup
    log_dir = './output/' + args.run_name + '/logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=log_dir + 'run.log',
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.setLevel(logging.INFO)
    tlogging.set_verbosity(logging.INFO)

    # Create dir
    results_dir = './output/' + args.run_name + '/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # ~~ DATA GENERATION ~~
    train, validation, test, relations, precision_k, test_alias_lookup = generate_knowledge_transfer(
        source_language=args.source_language,
        target_language=args.target_language,
        n_relations=args.n_relations,
        n_facts=args.n_facts,
        use_alias=args.use_alias,
        evaluate_test=args.evaluate_test,
        multilingual_entities=args.multilingual,
        multilingual_object=args.multilingual_object,
        multilingual_subject=args.multilingual_subject,
        verify_model=args.verify_model,
        frequency_test=args.frequency_test,
        reuse_test=args.reuse_test,
        cs_test=args.cs_test,
        max_subject_per_relation=args.subject_per_relation,
        max_subject_all_relation=args.subject_all_relation,
        n_shot=args.n_shot,
        train_w_alias=args.train_w_alias,
        source_entities=args.source_entities,
        source_sov=args.source_sov,
        use_bert=args.bert,
        use_fixed_relations=args.use_fixed_relations,
        run_name=args.run_name
    )
    args.max_precision_k = max(precision_k.values())

    logger.info(f"Training Parameters {args}")

    # LOADING MODEL AND TOKENIZER
    if args.bert:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        if 'untrained' in args.run_name:
            # Untrained:
            config = BertConfig.from_pretrained("bert-base-cased")
            model = BertForMaskedLM(config)
        else:
            model = BertForMaskedLM.from_pretrained("bert-base-cased")
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

    # Load Data Collator for Prediction and Evaluation
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    eval_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ~~ PRE-PROCESSING ~~
    if args.dot_test:
        # Add dot to training
        for i, sample in enumerate(train):
            train[i] = sample + ' .'
    train_dict = {'sample': train}

    if args.frequency_test:
        # Validation is already just a list and not a dictionary
        # validation_dict = {'sample': validation}
        # test_dict = {'sample': flatten_dict3_w_key(copy.deepcopy(test), 'relation')}
        validation_list = flatten_dict_to_list(copy.deepcopy(validation), 'relation')
        test_list = flatten_dict_to_list(copy.deepcopy(test), 'relation')

        validation_dict = {'sample': validation_list}
        test_dict = {'sample': test_list}
    else:
        # Target is SOV not SVO
        if args.target_sov:
            validation_list = dict_to_list_sov(validation)
            test_list = flatten_dict_to_list_sov(copy.deepcopy(test), 'relation')
        else:
            validation_list = dict_to_list(validation)
            test_list = flatten_dict_to_list(copy.deepcopy(test), 'relation')

        if args.dot_test:
            # Add dot to training
            for i, sample in enumerate(validation_list):
                validation_list[i] = sample + ' .'
            for i, sample in enumerate(test_list):
                test_list[i] = sample + ' .'

        validation_dict = {'sample': validation_list}
        test_dict = {'sample': test_list}

    train_ds = Dataset.from_dict(train_dict)
    validation_ds = Dataset.from_dict(validation_dict)
    test_ds = Dataset.from_dict(test_dict)

    # Tokenize Training and Test Data
    tokenized_train = tokenize(tokenizer, train_ds)  # Train is shuffled by Huggingface
    tokenized_validation = tokenize(tokenizer, validation_ds)
    tokenized_test = tokenize(tokenizer, test_ds)

    # Tokenize Test Alias Lookup dict
    # max_len = 0
    if args.combined_metric:
        test_alias_lookup_tokenized = defaultdict(list)
        for fact in test_alias_lookup:

            fact_tokenized = str(tokenizer(fact)['input_ids'])

            # Tokenize alias facts in batches and pad them
            alias_fact_tokenized = tokenizer(test_alias_lookup[fact], return_tensors='pt', padding="max_length", max_length=15)
            # Add labels
            alias_fact_tokenized['labels'] = copy.deepcopy(alias_fact_tokenized['input_ids'])
            for i, idx1 in enumerate(alias_fact_tokenized['labels']):
                for j, idx2 in enumerate(idx1):
                    if idx2 == 0:
                        alias_fact_tokenized['labels'][i][j] = -100

            # new_max = find_max_list(alias_fact_tokenized['input_ids'])
            # if new_max > max_len:
            #     max_len = new_max

            test_alias_lookup_tokenized[fact_tokenized] = alias_fact_tokenized

    training_args = TrainingArguments(
        output_dir='./output/' + args.run_name + '/models/',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=128,
        learning_rate=args.lr,
        logging_dir='./output/' + args.run_name + '/tb_logs/',
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=120,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=120,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=120,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        run_name=args.run_name,
        seed=args.seed
    )

    # Use precision_at_k_fixed metric with fixed k
    if args.precision_k > 0:
        if args.max_precision_k > args.precision_k:
            raise ValueError('precision@k needs to be at least as large as needed by the dataset!')

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test if args.evaluate_test else tokenized_validation,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=precision_at_k_fixed_alias if args.combined_metric else precision_at_k_fixed,
            precision_at=args.precision_k,
            target_sov=args.target_sov,
            dot_test=args.dot_test,
            alias_lookup=test_alias_lookup_tokenized if args.combined_metric else None
        )
    elif args.metric_mrr:
        # Takes 1000 highest predictions and weighs the correct ones by their rank
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test if args.evaluate_test else tokenized_validation,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=mean_reciprocal_rank_alias if args.combined_metric else mean_reciprocal_rank,
            precision_at=args.max_precision_k,
            target_sov=args.target_sov,
            dot_test=args.dot_test,
            alias_lookup=test_alias_lookup_tokenized if args.combined_metric else None
        )
    # Use precision@k with k = amount of possible entities
    elif args.max_precision_k > 1:
        # Tokenize precision dict
        tokenized_precision_k = tokenize_dict_keys(precision_k, tokenizer)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test if args.evaluate_test else tokenized_validation,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=precision_at_k,
            precision_at=args.max_precision_k,
            precision_dict=tokenized_precision_k,
            target_sov=args.target_sov,
            dot_test=args.dot_test,
            alias_lookup=test_alias_lookup_tokenized if args.combined_metric else None
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test if args.evaluate_test else tokenized_validation,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=precision_at_one_alias if args.combined_metric else precision_at_one,
            target_sov=args.target_sov,
            dot_test=args.dot_test,
            alias_lookup=test_alias_lookup_tokenized if args.combined_metric else None
        )

    # Train
    logger.info("Training...")
    trainer.train()

    # Stop Tensorboard
    trainer.remove_callback(TensorBoardCallback)

    # Save Train, Validation and Test
    logger.info("Saving training, validation and test data")

    train_df = pd.DataFrame(train_dict)
    validation_df = pd.DataFrame(validation_dict)
    test_df = pd.DataFrame(test)

    data_dir = './output/' + args.run_name + '/data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_df.to_csv(data_dir + 'train_set')
    validation_df.to_csv(data_dir + 'validation_set')
    test_df.to_json(data_dir + 'test_set')

    # Evaluate zero-shot crosslingual transfer - test relation precision
    logger.info('')
    logger.info('Evaluate...')

    logger.info(f'Source: {args.source_language}, Target: {str(args.target_language)}')
    logger.info('')

    logger.info('**** TEST ****')
    logger.info('*******************')
    metrics = trainer.evaluate(eval_dataset=tokenized_test)
    eval_acc = float("{:.2f}".format(metrics['eval_accuracy']))
    logger.info(f'Test Accuracy: {eval_acc}')

    logger.info('**** RELATIONS ****')
    logger.info('*******************')
    evaluation_relation(trainer, tokenizer, relations, args, copy.deepcopy(test))

    if args.use_alias:
        logger.info('')
        logger.info('**** ALIAS ****')
        logger.info('***************')
        evaluation_alias(trainer, tokenizer, relations, args, copy.deepcopy(test), 'alias')

        logger.info('**** TRANSLATION ****')
        logger.info('*********************')
        evaluation_alias(trainer, tokenizer, relations, args, copy.deepcopy(test), 'translate')

        logger.info('**** SUBWORDS ****')
        logger.info('******************')
        evaluation_alias(trainer, tokenizer, relations, args, copy.deepcopy(test), 'subword')

    if args.frequency_test:
        logger.info('**** FREQUENCY ****')
        logger.info('******************')
        evaluation_frequency(trainer, tokenizer, relations, args, copy.deepcopy(test))

    logger.info('Done.')


def evaluation_frequency(trainer, tokenizer, relations, args, test):
    if 1 < len(args.source_language):
        raise NotImplementedError('Not implemented multiple source languages.')

    # Prepare test set like in relation
    test_relations = flatten_remove_dict(copy.deepcopy(test), 'relation')

    # Create dataframe for every relation
    frequency_acc_per_relation = {relation[args.source_language[0]]: pd.DataFrame()
                                  for _, relation in relations.iterrows()}

    split_size = int(args.n_facts / 4)
    eval_split_size = 10

    # List of frequencies
    freqs = [1, 10, 50, 100]

    for _, relation in relations.iterrows():

        df = frequency_acc_per_relation[relation[args.source_language[0]]]

        acc = defaultdict(list)

        # For every language compute the accuracy of every frequency bucket
        for test_lang in args.target_language:
            logger.info(f'LANGUAGE: {test_lang}')

            for i, freq in enumerate(freqs):
                # Take test set in split steps
                relation_test = test_relations[test_lang][relation[test_lang]][
                                i * (split_size - eval_split_size):(i + 1) * (split_size - eval_split_size)]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds)
                print(metrics)

                # Save accuracy
                eval_acc = float("{:.2f}".format(metrics['eval_accuracy']))
                acc[test_lang].append(eval_acc)

                if args.use_alias:
                    # Alias, Translation and Subwords
                    alt_names = ['alias', 'translate', 'subword']

                    for alt_name in alt_names:
                        for alias in test[test_lang][relation[test_lang]][alt_name]:
                            alias_test = test[test_lang][relation[test_lang]][alt_name][alias][
                                         i * split_size:(i + 1) * split_size]
                            relation_test_ds = Dataset.from_dict({'sample': alias_test})
                            tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)
                            metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds)
                            eval_acc = float("{:.2f}".format(metrics['eval_accuracy']))
                            acc[test_lang + '_' + alt_name + '_' + alias].append(eval_acc)

        df['frequency'] = freqs

        for test_lang in args.target_language:
            df['acc_' + test_lang] = acc[test_lang]

            if args.use_alias:
                alt_names = ['alias', 'translate', 'subword']

                for alt_name in alt_names:
                    for alias in test[test_lang][relation[test_lang]][alt_name]:
                        df['acc_' + test_lang + '_' + alt_name + '_' + alias] = acc[
                            test_lang + '_' + alt_name + '_' + alias]

    # SAVING RESULTS
    results_dir = './output/' + args.run_name + '/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for relation in frequency_acc_per_relation:
        frequency_acc_per_relation[relation].to_csv(
            './output/' + args.run_name + '/results/frequency_' + relation + '.csv', index=False)


def evaluation_alias(trainer, tokenizer, relations, args, test, test_key='relation'):
    double_print = False

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
    if args.target_sov:
        test = test_to_normal_alias_sov(test, test_key)
    else:
        test = test_to_normal_alias(test, test_key)
    test_relations = flatten_remove_dict4(test, test_key)

    for language_code in test.keys():
        language_ds = Dataset.from_dict({language_code: test_relations[language_code]})
        test_datasets.append(language_ds)

    # Iterate over relations to evaluate
    for _, relation in relations.iterrows():
        for source in args.source_language:
            logger.info(f'RELATION: {relation[source]}')

        # Iterate over all relations per target language
        for test_lang in args.target_language:
            logger.info(f'-> LANGUAGE: {test_lang} - TARGET: {relation[test_lang]}')

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

                for source in args.source_language:
                    token_set_source = set(tokenizer(relation[source])['input_ids'][1:-1])
                    token_set_target = set(tokenizer(alias)['input_ids'][1:-1])
                    logger.info(f'Similarity Measures - Relation - source: {relation[source]}, target: {alias}')
                    logger.info(f'Overlap Coefficient: {overlap_coefficient(token_set_source, token_set_target)}')
                    logger.info(f'Jaccard Index: {jaccard_index(token_set_source, token_set_target)}')

            logger.info('')
            double_print = True

        if not double_print:
            logger.info('')
        double_print = False


def evaluation_relation(trainer, tokenizer, relations, args, test):
    # Create separate language Datasets for evaluation
    test_datasets = []

    # Test needs to have all facts as list but
    if args.target_sov:
        test = test_to_normal_sov(test)
    else:
        test = test_to_normal(test)

    test_relations = flatten_remove_dict(test, 'relation')

    for language_code in test.keys():
        language_ds = Dataset.from_dict({language_code: test_relations[language_code]})
        test_datasets.append(language_ds)

    # Dict of dataframe per relation in every language with all entities, key is in source lang
    entity_acc_per_relation = {relation[source]: pd.DataFrame() for source in args.source_language for _, relation in
                               relations.iterrows()}
    entity_acc_per_language = {language_code: pd.DataFrame() for language_code in test}

    # Iterate over target languages to evaluate
    for test_lang in args.target_language:
        logger.info(f'LANGUAGE: {test_lang}')

        # Dictionary {entityA: {relationA: correct/total, relationB: correct/total}, entityB: {...}}
        entity_relation_acc = {}

        # Iterate over all relations per target language
        for _, relation in relations.iterrows():
            for source in args.source_language:
                logger.info(f'Relation - source: {relation[source]}, target: {relation[test_lang]}')

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

            # Compute similarity measures
            for source in args.source_language:
                token_set_source = set(tokenizer(relation[source])['input_ids'][1:-1])
                token_set_target = set(tokenizer(relation[test_lang])['input_ids'][1:-1])
                logger.info(
                    f'Similarity Measures - Relation - source: {relation[source]}, target: {relation[test_lang]}')
                logger.info(f'Overlap Coefficient: {overlap_coefficient(token_set_source, token_set_target)}')
                logger.info(f'Jaccard Index: {jaccard_index(token_set_source, token_set_target)}')

            # Compute number of occurences for this relation
            entity_occurences = compute_occurences(relation_test, metrics, args.target_sov)

            # *** DATAFRAME PER LANGUAGE FOR EVERY RELATION WITH ALL ENTITIES ***
            for entity in entity_occurences:
                if entity not in entity_relation_acc:
                    entity_relation_acc[entity] = {}

                # Compute the correctly predited ratio for every entity appearing in this relation
                correct = entity_occurences[entity]['correct_subject'] + entity_occurences[entity]['correct_object']
                total = entity_occurences[entity]['subject'] + entity_occurences[entity]['object']

                for source in args.source_language:
                    entity_relation_acc[entity][relation[source]] = correct / total

            # *** DATAFRAME PER RELATION IN EVERY LANGUAGE WITH ALL ENTITIES ***
            # If there is no column entities? Create it
            for source in args.source_language:
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

        logger.info('')

        # Entity/Relation Accuracy Dict to Dataframe (and Transpose so that entities are on rows)
        entity_acc_per_language[test_lang] = pd.DataFrame(entity_relation_acc).T

    # Save entity results
    results_dir = './output/' + args.run_name + '/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for relation in entity_acc_per_relation:
        entity_acc_per_relation[relation].to_csv('./output/' + args.run_name + '/results/' + relation + '.csv',
                                                 index=False)

    for language in entity_acc_per_language:
        entity_acc_per_language[language].to_csv('./output/' + args.run_name + '/results/' + language + '.csv',
                                                 index=False)


def compute_occurences(relation_test, metrics, target_sov):
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
