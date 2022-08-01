#!/bin/bash
# Multilingual European - Single Source
#python run_knowledge.py -r KTm_en_de-fr-es -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual
#python run_knowledge.py -r KTm_de_en-fr-es -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual
#python run_knowledge.py -r KTm_fr_de-en-es -s fr -t de en es --lr 6e-5 --batch_size 200 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual
#python run_knowledge.py -r KTm_es_de-en-fr -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual

# Multilingual European - Single Source - Language-pairs
#python run_knowledge.py -r KTm_en_de -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_en_fr -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_en_es -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test

#python run_knowledge.py -r KTm_de_en -s de -t en --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_de_fr -s de -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_de_es -s de -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test

#python run_knowledge.py -r KTm_fr_de -s fr -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_fr_en -s fr -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_fr_es -s fr -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test

#python run_knowledge.py -r KTm_es_de -s es -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_es_en -s es -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_es_fr -s es -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test

# Multilingual European - Single Source - Train with Alias
#python run_knowledge.py -r KTm_en_de-fr-es_trainalias -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias
#python run_knowledge.py -r KTm_de_en-fr-es_trainalias -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias
#python run_knowledge.py -r KTm_fr_de-en-es_trainalias -s fr -t de en es --lr 6e-5 --batch_size 200 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias
#python run_knowledge.py -r KTm_es_de-en-fr_trainalias -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias

# Multilingual European - Single Source - Test with Alias
#python run_knowledge.py -r KTm_en_de-fr-es_testalias -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias
#python run_knowledge.py -r KTm_de_en-fr-es_testalias -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias
#python run_knowledge.py -r KTm_fr_de-en-es_testalias -s fr -t de en es --lr 6e-5 --batch_size 200 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias
#python run_knowledge.py -r KTm_es_de-en-fr_testalias -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias

# Multilingual European - Single Source - Train + Test with Alias
#python run_knowledge.py -r KTm_en_de-fr-es -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --combined_metric --group TrainTestAlias
#python run_knowledge.py -r KTm_de_en-fr-es -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --combined_metric --group TrainTestAlias
#python run_knowledge.py -r KTm_fr_de-en-es -s fr -t de en es --lr 6e-5 --batch_size 200 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --combined_metric --group TrainTestAlias
#python run_knowledge.py -r KTm_es_de-en-fr -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --combined_metric --group TrainTestAlias

# Multilingual European - Single Source - Agnostic Subject + Multilingual Object
#python run_knowledge.py -r KTm_en_de_object -s en -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_en_fr_object -s en -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_en_es_object -s en -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test

#python run_knowledge.py -r KTm_de_en_object -s de -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_de_fr_object -s de -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_de_es_object -s de -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test

#python run_knowledge.py -r KTm_fr_de_object -s fr -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_fr_en_object -s fr -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_fr_es_object -s fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test

#python run_knowledge.py -r KTm_es_de_object -s es -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_es_en_object -s es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_es_fr_object -s es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test

# Multilingual European - Single Source - Multilingual Subject + Agnostic Object
#python run_knowledge.py -r KTm_en_de_subject -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_en_fr_subject -s en -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_en_es_subject -s en -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test

#python run_knowledge.py -r KTm_de_en_subject -s de -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_de_fr_subject -s de -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_de_es_subject -s de -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test

#python run_knowledge.py -r KTm_fr_de_subject -s fr -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_fr_en_subject -s fr -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_fr_es_subject -s fr -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test

#python run_knowledge.py -r KTm_es_de_subject -s es -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_es_en_subject -s es -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_es_fr_subject -s es -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test

# Multilingual European - Single Source - English-to-French w/ alternative scores
python run_knowledge.py -r KTm_en_fr_mrr_test -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --metric_mrr --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_fr_precision_at_10_test -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 10 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_fr_precision_at_50_test -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 50 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_fr_precision_at_100_test -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 100 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_fr_precision_at_200_test -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 200 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_fr_precision_at_500_test -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 500 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_fr_precision_at_1000_test -s en -t fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 1000 --evaluate_test --group alt_score

python run_knowledge.py -r KTm_en_de_mrr_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --metric_mrr --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_de_precision_at_10_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 10 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_de_precision_at_50_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 50 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_de_precision_at_100_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 100 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_de_precision_at_200_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 200 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_de_precision_at_500_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 500 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_de_precision_at_1000_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 1000 --evaluate_test --group alt_score

python run_knowledge.py -r KTm_en_es_mrr_test -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --metric_mrr --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_es_precision_at_10_test -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 10 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_es_precision_at_50_test -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 50 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_es_precision_at_100_test -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 100 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_es_precision_at_200_test -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 200 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_es_precision_at_500_test -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 500 --evaluate_test --group alt_score
python run_knowledge.py -r KTm_en_es_precision_at_1000_test -s en -t es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --precision_k 1000 --evaluate_test --group alt_score


# Multilingual European - Single Source - English-to-French w/ reduced entities and relations
#python run_knowledge.py -r KTm_en_fr_10r_500f -s en -t fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 500 --multilingual
#python run_knowledge.py -r KTm_en_fr_5r_500f -s en -t fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 500 --multilingual
#python run_knowledge.py -r KTm_en_fr_5r_1000f -s en -t fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 1000 --multilingual

#python run_knowledge.py -r KTm_fr_en_10r_500f -s fr -t en --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 500 --multilingual
#python run_knowledge.py -r KTm_fr_en_5r_500f -s fr -t en --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 500 --multilingual
#python run_knowledge.py -r KTm_fr_en_5r_1000f -s fr -t en --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 1000 --multilingual

# Multilingual European - Single Source - Dot Test
#python run_knowledge.py -r KTm_en_de-fr-es_dot -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test
#python run_knowledge.py -r KTm_de_en-fr-es_dot -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test
#python run_knowledge.py -r KTm_fr_de-en-es_dot -s fr -t de en es --lr 6e-5 --batch_size 200 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test
#python run_knowledge.py -r KTm_es_de-en-fr_dot -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test

# Multilingual Asian - Single Source
#python run_knowledge.py -r KTm_en_zh -s en -t zh --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --group mAsian
#python run_knowledge.py -r KTm_en_ja -s en -t ja --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --group mAsian
#python run_knowledge.py -r KTm_en_ru -s en -t ru --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --group mAsian
#python run_knowledge.py -r KTm_ru_ja -s ru -t ja --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --group mAsian
#python run_knowledge.py -r KTm_ru_zh -s ru -t zh --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --group mAsian
#python run_knowledge.py -r KTm_zh_ja -s zh -t ja --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual --group mAsian

# Multilingual Asian - Single Source - Japanese as SOV
#python run_knowledge.py -r KTm_en_ja_sov -s en -t ja --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --multilingual

# Multilingual European - Single Source - Validation: n_facts
#python run_knowledge.py -r KTm_en_de-fr-es_10r_500f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 500 --multilingual
#python run_knowledge.py -r KTm_en_de-fr-es_10r_250f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 250 --multilingual
#python run_knowledge.py -r KTm_en_de-fr-es_5r_500f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 500 --multilingual
#python run_knowledge.py -r KTm_en_de-fr-es_5r_1000f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 1000 --multilingual

#### MULTIPLE SOURCE LANGUAGES
# Multilingual European - Multiple Source Languages
#python run_knowledge.py -r KTm_de-fr-es_en_test -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_en-de-fr_es_test -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_en-fr-es_de_test -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_en-de-es_fr_test -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test

# Multilingual European - Multiple Source Languages - Dot Test
#python run_knowledge.py -r KTm_de-fr-es_en_dot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test
#python run_knowledge.py -r KTm_en-de-fr_es_dot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test
#python run_knowledge.py -r KTm_en-fr-es_de_dot -s en fr es -t de --lr 6e-5 --batch_size 200 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test
#python run_knowledge.py -r KTm_en-de-es_fr_dot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --dot_test --group KTm_dot_test

# Agnostic Subject + Multilingual Object
#python run_knowledge.py -r KTm_de-fr-es_en_object_test -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_en-de-fr_es_object_test -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_en-fr-es_de_object_test -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test
#python run_knowledge.py -r KTm_en-de-es_fr_object_test -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --multilingual --multilingual_object --group mObject --evaluate_test

# Multilingual Subject + Agnostic Object
#python run_knowledge.py -r KTm_de-fr-es_en_subject_test -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_en-de-fr_es_subject_test -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_en-fr-es_de_subject_test -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test
#python run_knowledge.py -r KTm_en-de-es_fr_subject_test -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --multilingual --multilingual_subject --group mSubject --evaluate_test

# Multilingual European - Multiple Source Languages - Train with Alias
#python run_knowledge.py -r KTm_de-fr-es_en_trainalias -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias
#python run_knowledge.py -r KTm_en-de-fr_es_trainalias -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias
#python run_knowledge.py -r KTm_en-fr-es_de_trainalias -s en fr es -t de --lr 6e-5 --batch_size 200 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias
#python run_knowledge.py -r KTm_en-de-es_fr_trainalias -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --train_w_alias --group train_w_alias

# Multilingual European - Multiple Source Languages - Test with Alias
#python run_knowledge.py -r KTm_de-fr-es_en_testalias -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias
#python run_knowledge.py -r KTm_en-de-fr_es_testalias -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias
#python run_knowledge.py -r KTm_en-fr-es_de_testalias -s en fr es -t de --lr 6e-5 --batch_size 200 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias
#python run_knowledge.py -r KTm_en-de-es_fr_testalias -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --combined_metric --group TestAlias

# Multilingual European - Multiple Source Languages - Validation: n_relations, n_facts
#python run_knowledge.py -r KTm_de-fr-es_en_10r_500f -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 500 --multilingual
#python run_knowledge.py -r KTm_de-fr-es_en_10r_250f -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 250 --multilingual
#python run_knowledge.py -r KTm_de-fr-es_en_5r_1000f -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 5 --n_facts 1000 --multilingual
#python run_knowledge.py -r KTm_de-fr-es_en_1r_1000f -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 1 --n_facts 1000 --multilingual
#python run_knowledge.py -r KTm_de-fr-es_en_5r_500f -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 5 --n_facts 500 --multilingual

# Multilingual European - Multiple Source Languages - Few-Shot
#python run_knowledge.py -r KTm_de-fr-es_en_10shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 10 --group n_shot
#python run_knowledge.py -r KTm_de-fr-es_en_25shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 25 --group n_shot
#python run_knowledge.py -r KTm_de-fr-es_en_50shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 50 --group n_shot
#python run_knowledge.py -r KTm_de-fr-es_en_100shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 100 --group n_shot
#python run_knowledge.py -r KTm_de-fr-es_en_200shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 200 --group n_shot

#python run_knowledge.py -r KTm_en-fr-es_de_10shot -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 10 --group n_shot
#python run_knowledge.py -r KTm_en-fr-es_de_25shot -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 25 --group n_shot
#python run_knowledge.py -r KTm_en-fr-es_de_50shot -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 50 --group n_shot
#python run_knowledge.py -r KTm_en-fr-es_de_100shot -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 100 --group n_shot
#python run_knowledge.py -r KTm_en-fr-es_de_200shot -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 200 --group n_shot

#python run_knowledge.py -r KTm_en-de-es_fr_10shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 10 --group n_shot
#python run_knowledge.py -r KTm_en-de-es_fr_25shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 25 --group n_shot
#python run_knowledge.py -r KTm_en-de-es_fr_50shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 50 --group n_shot
#python run_knowledge.py -r KTm_en-de-es_fr_100shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 100 --group n_shot
#python run_knowledge.py -r KTm_en-de-es_fr_200shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 200 --group n_shot

#python run_knowledge.py -r KTm_en-de-fr_es_10shot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 10 --group n_shot
#python run_knowledge.py -r KTm_en-de-fr_es_25shot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 25 --group n_shot
#python run_knowledge.py -r KTm_en-de-fr_es_50shot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 50 --group n_shot
#python run_knowledge.py -r KTm_en-de-fr_es_100shot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 100 --group n_shot
#python run_knowledge.py -r KTm_en-de-fr_es_200shot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --n_shot 200 --group n_shot

# Multilingual European - Multiple Source Languages - Alt metric MRR
#python run_knowledge.py -r KTm_de-fr-es_en_mrr -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --metric_mrr --group alt_score
#python run_knowledge.py -r KTm_en-de-fr_es_mrr -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations w10 --n_facts 1000 --multilingual --metric_mrr --group alt_score
#python run_knowledge.py -r KTm_en-fr-es_de_mrr -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --metric_mrr --group alt_score
#python run_knowledge.py -r KTm_en-de-es_fr_mrr -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --multilingual --metric_mrr --group alt_score

# Multilingual European - Multiple Source Languages - Alt metric Precision@K
#python run_knowledge.py -r KTm_de-fr-es_en_p10 -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --precision_k 10 --evaluate_test --group alt_score
#python run_knowledge.py -r KTm_de-fr-es_en_p20 -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --precision_k 20 --evaluate_test --group alt_score
#python run_knowledge.py -r KTm_de-fr-es_en_p50 -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --precision_k 50 --evaluate_test --group alt_score
#python run_knowledge.py -r KTm_de-fr-es_en_p100 -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --precision_k 100 --evaluate_test --group alt_score
#python run_knowledge.py -r KTm_en-de-fr_es_p10 -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --precision_k 10 --group alt_score
#python run_knowledge.py -r KTm_en-fr-es_de_p10 -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --precision_k 10 --group alt_score
#python run_knowledge.py -r KTm_en-de-es_fr_p10 -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --multilingual --precision_k 10 --group alt_score

# Combining everything
#python run_knowledge.py -r KTm_de-fr-es_en_5r_p10_50shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 5 --n_facts 1000 --multilingual --precision_k 10 --n_shot 50 --evaluate_test
#python run_knowledge.py -r KTm_en-fr-es_de_5r_p10_50shot -s en fr es -t de --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 5 --n_facts 1000 --multilingual --precision_k 10 --n_shot 50 --evaluate_test
#python run_knowledge.py -r KTm_en-de-es_fr_5r_p10_50shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 5 --n_facts 1000 --multilingual --precision_k 10 --n_shot 50 --evaluate_test
#python run_knowledge.py -r KTm_en-de-fr_es_5r_p10_50shot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 5 --n_facts 1000 --multilingual --precision_k 10 --n_shot 50 --evaluate_test
