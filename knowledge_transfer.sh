#!/bin/bash
# 1 Source language
####################
#python run_knowledge.py -r KT_en_de-fr-es -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000
#python run_knowledge.py -r KT_de_en-fr-es -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000
#python run_knowledge.py -r KT_fr_de-en-es -s fr -t de en es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000
#python run_knowledge.py -r KT_es_de-en-fr -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000

#python run_knowledge.py -r KT_en_de -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_en_fr -s en -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_en_es -s en -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

#python run_knowledge.py -r KT_en_ru -s en -t ru --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_en_ja -s en -t ja --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_en_zh -s en -t zh --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

#python run_knowledge.py -r KT_de_en -s de -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_de_fr -s de -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_de_es -s de -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

#python run_knowledge.py -r KT_fr_de -s fr -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_fr_en -s fr -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_fr_es -s fr -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

#python run_knowledge.py -r KT_es_de -s es -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_es_en -s es -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_es_fr -s es -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

#python run_knowledge.py -r KT_en_ru -s en -t ru --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_en_ja -s en -t ja --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_en_zh -s en -t zh --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

# Non-latin languages w/ Latin entities
#python run_knowledge.py -r KT_ru_ja -s ru -t ja --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_ru_zh -s ru -t zh --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

#python run_knowledge.py -r KT_ja_ru -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_ja_zh -s ja -t zh --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

#python run_knowledge.py -r KT_zh_ru -s zh -t ru --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source
#python run_knowledge.py -r KT_zh_ja -s zh -t ja --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_1source

# Non-latin languages w/ Source entities
#python run_knowledge.py -r KT_ru_ja_source -s ru -t ja --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --source_entities --multilingual --group KT_1source
#python run_knowledge.py -r KT_ru_zh_source -s ru -t zh --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test --source_entities --multilingual --group KT_1source

#python run_knowledge.py -r KT_ja_ru_source -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --source_entities --multilingual --group KT_1source
#python run_knowledge.py -r KT_ja_zh_source -s ja -t zh --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --source_entities --multilingual --group KT_1source

#python run_knowledge.py -r KT_zh_ru_source -s zh -t ru --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --source_entities --multilingual --group KT_1source
#python run_knowledge.py -r KT_zh_ja_source -s zh -t ja --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --evaluate_test --source_entities --multilingual --group KT_1source

# Train with Alias
#python run_knowledge.py -r KT_en_de-fr-es_trainalias -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias
#python run_knowledge.py -r KT_de_en-fr-es_trainalias -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias
#python run_knowledge.py -r KT_fr_de-en-es_trainalias -s fr -t de en es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias
#python run_knowledge.py -r KT_es_de-en-fr_trainalias -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias

# Test with Alias
#python run_knowledge.py -r KT_en_de_testalias -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_en_fr_testalias -s en -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_en_es_testalias -s en -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias

#python run_knowledge.py -r KT_de_en_testalias -s de -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_de_fr_testalias -s de -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_de_es_testalias -s de -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias

#python run_knowledge.py -r KT_fr_de_testalias -s fr -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_fr_en_testalias -s fr -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_fr_es_testalias -s fr -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias

#python run_knowledge.py -r KT_es_de_testalias -s es -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_es_en_testalias -s es -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_es_fr_testalias -s es -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias

#python run_knowledge.py -r KT_ru_ja_testalias -s ru -t ja --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_ru_zh_testalias -s ru -t zh --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias

#python run_knowledge.py -r KT_ja_ru_testalias -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_ja_zh_testalias -s ja -t zh --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias

#python run_knowledge.py -r KT_zh_ru_testalias -s zh -t ru --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias
#python run_knowledge.py -r KT_zh_ja_testalias -s zh -t ja --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --combined_metric --evaluate_test --group KT_TestAlias

# Dot Test
#python run_knowledge.py -r KT_en_de_dot -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_en_fr_dot -s en -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_en_es_dot -s en -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test

#python run_knowledge.py -r KT_de_en_dot -s de -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_de_fr_dot -s de -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_de_es_dot -s de -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test

#python run_knowledge.py -r KT_fr_de_dot -s fr -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_fr_en_dot -s fr -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_fr_es_dot -s fr -t es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test

#python run_knowledge.py -r KT_es_de_dot -s es -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_es_en_dot -s es -t en --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test
#python run_knowledge.py -r KT_es_fr_dot -s es -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --dot_test --group KT_dot_test --evaluate_test

# SOV - agnostic
#python run_knowledge.py -r KT_en_ja_source_sov -s en -t ja --lr 5e-5 --batch_size 256 --epochs 300 --n_relations 10 --n_facts 1000 --target_sov --group SOV
#python run_knowledge.py -r KT_en_de_source_sov -s en -t de --lr 5e-5 --batch_size 256 --epochs 300 --n_relations 10 --n_facts 1000 --target_sov --group SOV
#python run_knowledge.py -r KT_en_ja_source_target_sov_test -s en -t ja --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --source_sov --target_sov --evaluate_test --group SOV
#python run_knowledge.py -r KT_en_de_source_target_sov_test -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --source_sov --target_sov --evaluate_test --group SOV

# Alt Metrics - MRR
#python run_knowledge.py -r KT_en_de-fr-es_mrr -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --metric_mrr --group alt_score
#python run_knowledge.py -r KT_de_en-fr-es_mrr -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --metric_mrr --group alt_score

# Alt Metrics - P@10
#python run_knowledge.py -r KT_en_de-fr-es_p10 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --precision_k 10 --group alt_score
#python run_knowledge.py -r KT_de_en-fr-es_p10 -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --precision_k 10 --group alt_score

# Reduction of Relation and Entities
#python run_knowledge.py -r KT_en_de-fr-es_10r_500f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 500
#python run_knowledge.py -r KT_en_de-fr-es_5r_1000f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 1000
#python run_knowledge.py -r KT_en_de-fr-es_5r_500f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 400 --n_relations 5 --n_facts 500

# 2 Source languages
####################
#python run_knowledge.py -r KT_en-de_fr_test -s en de -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source
#python run_knowledge.py -r KT_en-de_es_test -s en de -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source

#python run_knowledge.py -r KT_en-fr_de_test -s en fr -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source
#python run_knowledge.py -r KT_en-fr_es_test -s en fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source

#python run_knowledge.py -r KT_en-es_fr_test -s en es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source
#python run_knowledge.py -r KT_en-es_de_test -s en es -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source

#python run_knowledge.py -r KT_de-fr_en_test -s de fr -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source
#python run_knowledge.py -r KT_de-fr_es_test -s de fr -t es --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source

#python run_knowledge.py -r KT_de-es_en_test -s de es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source
#python run_knowledge.py -r KT_de-es_fr_test -s de es -t fr --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source

#python run_knowledge.py -r KT_fr-es_de_test -s fr es -t de --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source
#python run_knowledge.py -r KT_fr-es_en_test -s fr es -t en --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_2source

# 3 Source languages
####################
#python run_knowledge.py -r KT_en-de-fr_es_test -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_3source
#python run_knowledge.py -r KT_en-de-es_fr_test -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_3source
#python run_knowledge.py -r KT_en-es-fr_de_test -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_3source
#python run_knowledge.py -r KT_de-fr-es_en_test -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --evaluate_test --group KT_3source

# Dot Test
#python run_knowledge.py -r KT_en-de-fr_es_dot -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --dot_test --evaluate_test --group KT_dot_test
#python run_knowledge.py -r KT_en-de-es_fr_dot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --dot_test --evaluate_test --group KT_dot_test
#python run_knowledge.py -r KT_en-es-fr_de_dot -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --dot_test --evaluate_test --group KT_dot_test
#python run_knowledge.py -r KT_de-fr-es_en_dot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --dot_test --evaluate_test --group KT_dot_test

# Train w/ Aliases
#python run_knowledge.py -r KT_en-de-fr_es_trainalias -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 50 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias
#python run_knowledge.py -r KT_en-de-es_fr_trainalias -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 50 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias
#python run_knowledge.py -r KT_en-es-fr_de_trainalias -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 50 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias
#python run_knowledge.py -r KT_de-fr-es_en_trainalias -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 50 --n_relations 10 --n_facts 1000 --train_w_alias --group KT_train_w_alias

# Test w/ Aliases
#python run_knowledge.py -r KT_en-de-fr_es_testalias -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --combined_metric --group KT_TestAlias
#python run_knowledge.py -r KT_en-de-es_fr_testalias -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --combined_metric --group KT_TestAlias
#python run_knowledge.py -r KT_en-es-fr_de_testalias -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --combined_metric --group KT_TestAlias
#python run_knowledge.py -r KT_de-fr-es_en_testalias -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --combined_metric --group KT_TestAlias

# Alt Metric - MRR
#python run_knowledge.py -r KT_en-de-fr_es_mrr -s en de fr -t es --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --metric_mrr --group alt_score
#python run_knowledge.py -r KT_en-de-es_fr_mrr -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --metric_mrr --group alt_score
#python run_knowledge.py -r KT_en-es-fr_de_mrr -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --metric_mrr --group alt_score
#python run_knowledge.py -r KT_de-fr-es_en_mrr -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --metric_mrr --group alt_score

# Alt Metric - P@k
#python run_knowledge.py -r KT_en-es-fr_de_p10 -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --precision_k 10 --group alt_score
#python run_knowledge.py -r KT_en-es-fr_de_p50 -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --precision_k 50 --group alt_score
#python run_knowledge.py -r KT_en-es-fr_de_p100 -s en es fr -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --precision_k 100 --group alt_score
#python run_knowledge.py -r KT_de-fr-es_en_p10 -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --precision_k 10 --group alt_score
#python run_knowledge.py -r KT_de-fr-es_en_p50 -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --precision_k 50 --group alt_score
#python run_knowledge.py -r KT_de-fr-es_en_p100 -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --precision_k 100 --group alt_score
