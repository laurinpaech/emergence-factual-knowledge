import argparse
import copy

from transformers import BertForMaskedLM, BertTokenizerFast, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, IntervalStrategy
from datasets import Dataset
import os

from data_generation_reasoning import *
from util.utils import *
from custom_trainer import CustomTrainer
from datasets import load_metric
import logging
from transformers import logging as tlogging
import sys
from util.utils import set_seed
from transformers.integrations import WandbCallback, TensorBoardCallback

logger = logging.getLogger(__name__)


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


def main():
    # PARAMETERS
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '-r', type=str, default="RelationDefault",
                        help='How to name current run')
    parser.add_argument('--source_language', '-s', nargs='+', default=["en"], help='source language')
    parser.add_argument('--target_language', '-t', nargs='+', default=["de"], help='target language')
    parser.add_argument('--relation', type=str, default='equivalence')
    parser.add_argument('--n_relations', type=int, default='10')
    parser.add_argument('--n_facts', type=int, default='1000')
    parser.add_argument('--epochs', type=int, default='200', help='Default is 200 epochs')
    parser.add_argument('--batch_size', type=int, default='256')
    parser.add_argument('--lr', type=float, default='5e-5')
    parser.add_argument('--precision_k', type=int, default='1')
    parser.add_argument('--n_pairs', type=int, default='0')
    parser.add_argument("--generate_random", action='store_true')
    parser.add_argument("--use_pretrained", action='store_true',
                        help='Tests on already trained relations and if they can learn to transfer for new entities.')
    parser.add_argument("--use_random", action='store_true')
    parser.add_argument("--use_anti", action='store_true')
    parser.add_argument("--only_testfacts", action='store_true', help="Can be used to test impact of rule-specific "
                                                                      "relations")
    parser.add_argument("--use_enhanced", action='store_true',
                        help='COMP use facts to group entities or IMP use connected')
    parser.add_argument("--use_same_relations", action='store_true', help='COMP use same relations')
    parser.add_argument("--use_target", action='store_true',
                        help='Adds target samples to remove the need for transfer.')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--n_gpu", type=int, default=2, help="single or two gpus")
    args = parser.parse_args()

    # FIX RANDOMNESS
    set_seed(args.seed)

    # Log Relation Test and set wandb group and some default args
    if args.n_pairs == 0:
        # Default to 20 or less
        if 20 < int(0.9 * args.n_facts):
            args.n_pairs = 20
        else:
            args.n_pairs = int(0.9 * args.n_facts)

    if args.relation == 'equivalence':
        logger.info("EQUIVALENCE TEST")
    elif args.relation == 'symmetry':
        logger.info("SYMMETRY TEST")
    elif args.relation == 'inversion':
        logger.info("INVERSION TEST")
    elif args.relation == 'negation':
        logger.info("NEGATION TEST")
    elif args.relation == 'implication':
        logger.info("IMPLICATION TEST")
        logger.info(f'Number of Implications: {args.precision_k}')
    elif args.relation == 'composition':
        logger.info("COMPOSITION TEST")
    else:
        raise ValueError('UNKNOWN RELATION')

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

    # ~~ DATA GENERATION ~~
    train, test, relations = generate_reasoning(relation=Relation(args.relation),
                                                source_language=args.source_language,
                                                target_language=args.target_language,
                                                n_relations=args.n_relations,
                                                n_facts=args.n_facts,
                                                use_pretrained=args.use_pretrained,
                                                use_target=args.use_target,
                                                use_enhanced=args.use_enhanced,
                                                precision_k=args.precision_k,
                                                use_same_relations=args.use_same_relations,
                                                n_pairs=args.n_pairs)

    if args.use_random:
        # Generate half/half
        factor = 1.0
        n_random = factor * args.n_facts

        train_random, relations_random = generate_random(args.source_language, args.target_language, n_random,
                                                         args.n_relations)
        train += train_random

    if args.relation == 'symmetry' and args.use_anti:
        train_anti, test_anti, relations_anti = generate_anti(relations_symmetric=relations,
                                                              source_lang=args.source_language,
                                                              target_lang=args.target_language,
                                                              n_relations=args.n_relations,
                                                              n_facts=args.n_facts)
        train += train_anti

    logger.info(f"Training Parameters {args}")

    # LOADING
    # Load mBERT model and Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

    # Load Data Collator for Prediction and Evaluation
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    eval_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ~~ PRE-PROCESSING ~~
    if args.only_testfacts:
        train_testfacts = []
        for i in range(args.n_relations):
            train_testfacts += train[1800 + i * 1900:(i + 1) * 1900]  # Note that its hardcoded for 1000 n_facts
        train_dict = {'sample': train_testfacts}
    else:
        train_dict = {'sample': train}
    test_dict = {'sample': flatten_dict2_list(copy.deepcopy(test))}
    train_ds = Dataset.from_dict(train_dict)
    test_ds = Dataset.from_dict(test_dict)

    # Save Train and Test Data
    logger.info("Saving training, validation and test data")

    train_df = pd.DataFrame(train_dict)
    test_complete_df = pd.DataFrame(test)
    test_flat_df = pd.DataFrame(test_dict)

    data_dir = './output/' + args.run_name + '/data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_df.to_csv(data_dir + 'train_set', index=False)
    test_complete_df.to_json(data_dir + 'test_set_complete')
    test_flat_df.to_csv(data_dir + 'test_set', index=False)

    if args.use_random:
        train_random_df = pd.DataFrame({'sample': train_random})
        train_random_df.to_csv(data_dir + 'train_random', index=False)

    if args.use_anti:
        train_anti_df = pd.DataFrame({'sample': train_anti})
        test_anti_df = pd.DataFrame({'sample': test_anti})

        train_anti_df.to_csv(data_dir + 'train_anti_set', index=False)
        test_anti_df.to_json(data_dir + 'test_anti_set')

    # Tokenize Training and Test Data
    tokenized_train = tokenize(tokenizer, train_ds)  # Train is shuffled by Huggingface
    tokenized_test = tokenize(tokenizer, test_ds)

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

    if args.precision_k > 1:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=precision_at_k_fixed,
            precision_at=args.precision_k
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=precision_at_one
        )

    # Train
    logger.info("Training...")
    trainer.train()

    # Stop Tensorboard
    trainer.remove_callback(TensorBoardCallback)

    # Evaluate...
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

    logger.info(f'**** {args.relation.upper()} ****')
    logger.info('*********************')

    if args.relation == 'equivalence':
        if args.use_pretrained:
            evaluation_equivalence_pretrained(trainer, tokenizer, relations, args, copy.deepcopy(test))
        else:
            evaluation_equivalence(trainer, tokenizer, relations, args, copy.deepcopy(test))
    elif args.relation == 'symmetry':
        evaluation_symmetry(trainer, tokenizer, relations, args, copy.deepcopy(test))
        if args.use_anti:
            logger.info(f'**** ASYMMETRY ****')
            evaluation_symmetry(trainer, tokenizer, relations_anti, args, copy.deepcopy(test_anti))
    elif args.relation == 'inversion':
        if args.use_pretrained:
            evaluation_inversion_pretrained(trainer, tokenizer, relations, args, copy.deepcopy(test))
        else:
            evaluation_inversion(trainer, tokenizer, relations, args, copy.deepcopy(test))
    elif args.relation == 'negation':
        evaluation_negation(trainer, tokenizer, relations, args, copy.deepcopy(test))
    elif args.relation == 'implication':
        evaluation_implication(trainer, tokenizer, relations, args, copy.deepcopy(test))
    elif args.relation == 'composition':
        evaluation_composition(trainer, tokenizer, relations, args, copy.deepcopy(test))
    else:
        raise ValueError('Relation incorrect.')


def evaluation_equivalence_pretrained(trainer, tokenizer, relations, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            # RELATION
            for source in args.source_language:
                logger.info(f"Relation - source: {relation[source]} Relation - target: {relation[target]}")

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
            for source in args.source_language:
                logger.info(
                    f"Alias - source: {relation[source + '_alias']} Alias - target: {relation[target + '_alias']}")

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
            logger.info('')

        logger.info('')


def evaluation_equivalence(trainer, tokenizer, relation_pairs, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for (idx1, relation1), (idx2, relation2) in relation_pairs:
            # RELATION1
            for source in args.source_language:
                logger.info(f'Relation1 - source: {relation1[source]}, target: {relation1[target]}')

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
            for source in args.source_language:
                logger.info(f'Relation2 - source: {relation2[source]}, target: {relation2[target]}')

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
                logger.info('')

        logger.info('')


def evaluation_symmetry(trainer, tokenizer, relations, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            for source in args.source_language:
                logger.info(f'Relation - source: {relation[source]}, target: {relation[target]}')

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
            logger.info('')

        logger.info('')


def evaluation_inversion_pretrained(trainer, tokenizer, relations, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            # RELATION
            for source in args.source_language:
                logger.info(
                    f"Relation - source: {relation[source + str(1)]} Relation - target: {relation[target + str(1)]}")

            if test[target][relation[target + str(1)]]:
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
            for source in args.source_language:
                logger.info(
                    f"Inversion - source: {relation[source + str(2)]} Alias - target: {relation[target + str(2)]}")

            if test[target][relation[target + str(2)]]:
                # Relation from test set dict
                relation_test = test[target][relation[target + str(2)]]

                # Tokenize
                relation_test_ds = Dataset.from_dict({'sample': relation_test})
                tokenized_relation_ds = tokenize(tokenizer, relation_test_ds)

                # Evaluate
                metrics = trainer.evaluate(eval_dataset=tokenized_relation_ds, custom_eval=True)
                output_metrics = remove_key_dict(metrics, 'eval_correct_predictions')
                print(output_metrics)
            logger.info('')

        logger.info('')


def evaluation_inversion(trainer, tokenizer, relation_pairs, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for (idx1, relation1), (idx2, relation2) in relation_pairs:
            # RELATION
            for source in args.source_language:
                logger.info(f'Relation - source: {relation1[source]}, target: {relation1[target]}')

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
            for source in args.source_language:
                logger.info(f'Inversion - source: {relation2[source]}, target: {relation2[target]}')

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
                logger.info('')

        logger.info('')


def evaluation_negation(trainer, tokenizer, relations, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for idx, relation in relations.iterrows():
            # RELATION
            for source in args.source_language:
                logger.info(f"Relation - source: {relation[source]} Relation - target: {relation[target]}")

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
            for source in args.source_language:
                logger.info(
                    f"Negation - source: {relation[source + '_negated']} Alias - target: {relation[target + '_negated']}")

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
                logger.info('')

        logger.info('')


def evaluation_implication(trainer, tokenizer, relation_pairs, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for (idx1, relation), (idx2, implication) in relation_pairs:
            # IMPLICATION
            for source in args.source_language:
                logger.info(f'Relation - source: {relation[source]}, target: {relation[target]}')
                logger.info(f'Implication - source: {implication[source]}, target: {implication[target]}')

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

        logger.info('')


def evaluation_composition(trainer, tokenizer, relation_triples, args, test):
    for target in test:
        logger.info(f'LANGUAGE: {target}')

        # Iterate over all relations per target language
        for (idx1, relation1), (idx2, relation2), (idx3, composition) in relation_triples:
            # IMPLICATION
            for source in args.source_language:
                logger.info(f'Relation1 - source: {relation1[source]}, target: {relation1[target]}')
                logger.info(f'Relation2 - source: {relation2[source]}, target: {relation2[target]}')
                logger.info(f'Composition - source: {composition[source]}, target: {composition[target]}')

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

        logger.info('')


if __name__ == "__main__":
    main()
