#!/bin/bash
# Multilingual European - Single Source
#python run_knowledge.py -r KTm_en_de-fr-es_test -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_de_en-fr-es_test -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_fr_de-en-es_test -s fr -t de en es --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_es_de-en-fr_test -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 280 --n_relations 10 --n_facts 1000 --multilingual --evaluate_test

# 1 Source language
#python run_knowledge.py -r KT_en_de-fr-es_test -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test
#python run_knowledge.py -r KT_de_en-fr-es_test -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test
#python run_knowledge.py -r KT_fr_de-en-es_test -s fr -t de en es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test
#python run_knowledge.py -r KT_es_de-en-fr_test -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --evaluate_test

# 1 Source language - Fixed Relations to test if having different entities makes a difference or if relations perform good in general
#python run_knowledge.py -r KT_en_de-fr-es_test_fixed_seed69 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --seed 69 --evaluate_test --use_fixed_relations
#python run_knowledge.py -r KT_en_de-fr-es_test_fixed_seed10 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --seed 10 --evaluate_test --use_fixed_relations
