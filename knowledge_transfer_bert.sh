#!/bin/bash
# BERT KT
####################
#python run_knowledge.py -r KT_en_bert_valid -s en -t en --lr 5e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --bert --group BERT
#python run_knowledge.py -r KT_de_bert_valid -s de -t en --lr 5e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --bert --group BERT
#python run_knowledge.py -r KT_es_bert_valid -s es -t en --lr 5e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --bert --group BERT
#python run_knowledge.py -r KT_fr_bert_valid -s fr -t en --lr 5e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --bert --group BERT
#python run_knowledge.py -r KT_zh_bert_valid -s zh -t en --lr 5e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --bert --group BERT
#python run_knowledge.py -r KT_ru_bert_valid -s ru -t en --lr 5e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --bert --group BERT
#python run_knowledge.py -r KT_ja_bert_valid -s ja -t en --lr 5e-5 --batch_size 256 --epochs 400 --n_relations 10 --n_facts 1000 --bert --group BERT

#python run_knowledge.py -r KT_en_bert -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_de_bert -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_es_bert -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_fr_bert -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_zh_bert -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_ru_bert -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_ja_bert -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_en_bert -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test

#python run_knowledge.py -r KT_en_bert_seed10 -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_de_bert_seed10 -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_es_bert_seed10 -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_fr_bert_seed10 -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_zh_bert_seed10 -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_ru_bert_seed10 -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_ja_bert_seed10 -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_en_bert_seed10 -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test

#python run_knowledge.py -r KT_en_bert_seed69 -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_de_bert_seed69 -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_es_bert_seed69 -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_fr_bert_seed69 -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_zh_bert_seed69 -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_ru_bert_seed69 -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_ja_bert_seed69 -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_en_bert_seed69 -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test

#python run_knowledge.py -r KT_en_bert_untrained -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_de_bert_untrained -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_es_bert_untrained -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_fr_bert_untrained -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_ru_bert_untrained -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_zh_bert_untrained -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_ja_bert_untrained -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test

#python run_knowledge.py -r KT_en_bert_wo_overlap -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_de_bert_wo_overlap -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_es_bert_wo_overlap -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_fr_bert_wo_overlap -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_ru_bert_wo_overlap -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_zh_bert_wo_overlap -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test
#python run_knowledge.py -r KT_ja_bert_wo_overlap -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --evaluate_test

#python run_knowledge.py -r KT_en_bert_seed10_wo_overlap -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_de_bert_seed10_wo_overlap -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_es_bert_seed10_wo_overlap -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_fr_bert_seed10_wo_overlap -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_ru_bert_seed10_wo_overlap -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_zh_bert_seed10_wo_overlap -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_ja_bert_seed10_wo_overlap -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 10 --evaluate_test

#python run_knowledge.py -r KT_en_bert_seed69_wo_overlap -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_de_bert_seed69_wo_overlap -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_es_bert_seed69_wo_overlap -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_fr_bert_seed69_wo_overlap -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_ru_bert_seed69_wo_overlap -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_zh_bert_seed69_wo_overlap -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_ja_bert_seed69_wo_overlap -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --group BERT --seed 69 --evaluate_test

#python run_knowledge.py -r KT_en_bert_wo_overlap_testalias -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --evaluate_test
#python run_knowledge.py -r KT_de_bert_wo_overlap_testalias -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --evaluate_test
#python run_knowledge.py -r KT_es_bert_wo_overlap_testalias -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --evaluate_test
#python run_knowledge.py -r KT_fr_bert_wo_overlap_testalias -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --evaluate_test
#python run_knowledge.py -r KT_ru_bert_wo_overlap_testalias -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --evaluate_test
#python run_knowledge.py -r KT_zh_bert_wo_overlap_testalias -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --evaluate_test
#python run_knowledge.py -r KT_ja_bert_wo_overlap_testalias -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --evaluate_test

#python run_knowledge.py -r KT_en_bert_seed10_wo_overlap_testalias -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_de_bert_seed10_wo_overlap_testalias -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_es_bert_seed10_wo_overlap_testalias -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_fr_bert_seed10_wo_overlap_testalias -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_ru_bert_seed10_wo_overlap_testalias -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_zh_bert_seed10_wo_overlap_testalias -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 10 --evaluate_test
#python run_knowledge.py -r KT_ja_bert_seed10_wo_overlap_testalias -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 10 --evaluate_test

#python run_knowledge.py -r KT_en_bert_seed69_wo_overlap_testalias -s en -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_de_bert_seed69_wo_overlap_testalias -s de -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_es_bert_seed69_wo_overlap_testalias -s es -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_fr_bert_seed69_wo_overlap_testalias -s fr -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_ru_bert_seed69_wo_overlap_testalias -s ru -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_zh_bert_seed69_wo_overlap_testalias -s zh -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 69 --evaluate_test
#python run_knowledge.py -r KT_ja_bert_seed69_wo_overlap_testalias -s ja -t en --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --bert --combined_metric --group BERT --seed 69 --evaluate_test
