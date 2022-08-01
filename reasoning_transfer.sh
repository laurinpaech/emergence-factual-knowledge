#!/bin/bash
# Equivalence - 1 Source Language
####################
# Target
#python run_reasoning.py -r EQUI_en_de-fr-es_target -s en -t de fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_de_en-fr-es_target -s de -t en fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_fr_de-en-es_target -s fr -t de en es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_es_de-en-fr_target -s es -t de en fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence

#python run_reasoning.py -r EQUI_en_de_target -s en -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_en_fr_target -s en -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_en_es_target -s en -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_de_en_target -s de -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_de_fr_target -s de -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_de_es_target -s de -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_fr_de_target -s fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_fr_en_target -s fr -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_fr_es_target -s fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_es_de_target -s es -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_es_en_target -s es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_es_fr_target -s es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence

# Target + Pretrained
#python run_reasoning.py -r EQUI_en_de-fr-es_target_pretrained -s en -t de fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_en-fr-es_target_pretrained -s de -t en fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_de-en-es_target_pretrained -s fr -t de en es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_de-en-fr_target_pretrained -s es -t de en fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence

#python run_reasoning.py -r EQUI_en_de_target_pretrained -s en -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_fr_target_pretrained -s en -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_es_target_pretrained -s en -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_en_target_pretrained -s de -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_fr_target_pretrained -s de -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_es_target_pretrained -s de -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_de_target_pretrained -s fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_en_target_pretrained -s fr -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_es_target_pretrained -s fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_de_target_pretrained -s es -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_en_target_pretrained -s es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_fr_target_pretrained -s es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence

# Source
#python run_reasoning.py -r EQUI_en_de-fr-es -s en -t de fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_de_en-fr-es -s de -t en fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_fr_de-en-es -s fr -t de en es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_es_de-en-fr -s es -t de en fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence

#python run_reasoning.py -r EQUI_en_de -s en -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_en_fr -s en -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_en_es -s en -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_de_en -s de -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_de_fr -s de -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_de_es -s de -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_fr_de -s fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_fr_en -s fr -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_fr_es -s fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_es_de -s es -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_es_en -s es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_es_fr -s es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence

# Source + Pretrained
#python run_reasoning.py -r EQUI_en_de-fr-es_pretrained -s en -t de fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_en-fr-es_pretrained -s de -t en fr es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_de-en-es_pretrained -s fr -t de en es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_de-en-fr_pretrained -s es -t de en fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence

#python run_reasoning.py -r EQUI_en_de_pretrained -s en -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_fr_pretrained -s en -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_es_pretrained -s en -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_en_pretrained -s de -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_fr_pretrained -s de -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de_es_pretrained -s de -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_de_pretrained -s fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_en_pretrained -s fr -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_fr_es_pretrained -s fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_de_pretrained -s es -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_en_pretrained -s es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_es_fr_pretrained -s es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence

# General vs Pretrained
#python run_reasoning.py -r EQUI_en_de_target_pretrained_testfacts -s en -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --only_testfacts --group Equivalence
#python run_reasoning.py -r EQUI_es_en_target_pretrained_testfacts -s es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --only_testfacts --group Equivalence


# Equivalence - 3 Source Languages
####################
# Source
#python run_reasoning.py -r EQUI_en-de-fr_es -s en de fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_en-de-es_fr -s en de es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_en-es-fr_de -s en es fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_de-fr-es_en -s de fr es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Equivalence

# Target
#python run_reasoning.py -r EQUI_en-de-fr_es_target_longer -s en de fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_en-de-es_fr_target_longer -s en de es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_en-es-fr_de_target_longer -s en es fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_de-fr-es_en_target_longer -s de fr es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence

# Source + Pretrained
#python run_reasoning.py -r EQUI_en-de-fr_es_pretrained -s en de fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en-de-es_fr_pretrained -s en de es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en-es-fr_de_pretrained -s en es fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de-fr-es_en_pretrained -s de fr es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence

# Target + Pretrained
#python run_reasoning.py -r EQUI_en-de-fr_es_target_pretrained_longer -s en de fr -t es --relation equivalence --lr 4e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en-de-es_fr_target_pretrained_longer -s en de es -t fr --relation equivalence --lr 4e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en-es-fr_de_target_pretrained_longer -s en es fr -t de --relation equivalence --lr 4e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_de-fr-es_en_target_pretrained_longer -s de fr es -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence


# Symmetry - 1 Source Language
####################
# Target
#python run_reasoning.py -r SYM_en_de-fr-es_target_p10 -s en -t de fr es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --precision_k 10 --group Symmetry
#python run_reasoning.py -r SYM_de_en-fr-es_target_p10 -s de -t en fr es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --precision_k 10 --group Symmetry
#python run_reasoning.py -r SYM_fr_de-en-es_target_p10 -s fr -t de en es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --precision_k 10 --group Symmetry
#python run_reasoning.py -r SYM_es_de-en-fr_target_p10 -s es -t de en fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --precision_k 10 --group Symmetry

#python run_reasoning.py -r SYM_en_de_target -s en -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_en_fr_target -s en -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_en_es_target -s en -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_de_en_target -s de -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_de_fr_target -s de -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_de_es_target -s de -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_fr_de_target -s fr -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_fr_en_target -s fr -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_fr_es_target -s fr -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_es_de_target -s es -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_es_en_target -s es -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_es_fr_target -s es -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry

# Source
#python run_reasoning.py -r SYM_en_de-fr-es -s en -t de fr es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_de_en-fr-es -s de -t en fr es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_fr_de-en-es -s fr -t de en es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_es_de-en-fr -s es -t de en fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry

#python run_reasoning.py -r SYM_en_de -s en -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_en_fr -s en -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_en_es -s en -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_de_en -s de -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_de_fr -s de -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_de_es -s de -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_fr_de -s fr -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_fr_en -s fr -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_fr_es -s fr -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_es_de -s es -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_es_en -s es -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_es_fr -s es -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry

# Source + Pretrained
# Note: Not doing pairs because pretrained and general have equal performance.
#python run_reasoning.py -r SYM_en_de-fr-es_pretrained -s en -t de fr es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_de_en-fr-es_pretrained -s de -t en fr es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_fr_de-en-es_pretrained -s fr -t de en es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_es_de-en-fr_pretrained -s es -t de en fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry


# Symmetry - 3 Source Language
####################
# Source
#python run_reasoning.py -r SYM_en-de-fr_es -s en de fr -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_en-de-es_fr -s en de es -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_en-es-fr_de -s en es fr -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_de-fr-es_en -s de fr es -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --group Symmetry

# Source + Pretrained
#python run_reasoning.py -r SYM_en-de-fr_es_pretrained -s en de fr -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_en-de-es_fr_pretrained -s en de es -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_en-es-fr_de_pretrained -s en es fr -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_de-fr-es_en_pretrained -s de fr es -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry

# Target
#python run_reasoning.py -r SYM_en-de-fr_es_target_longer -s en de fr -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_en-de-es_fr_target_longer -s en de es -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_en-es-fr_de_target_longer -s en es fr -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_de-fr-es_en_target_longer -s de fr es -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Symmetry

# Target + Pretrained
#python run_reasoning.py -r SYM_en-de-fr_es_target_pretrained -s en de fr -t es --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_en-de-es_fr_target_pretrained -s en de es -t fr --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_en-es-fr_de_target_pretrained -s en es fr -t de --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_de-fr-es_en_target_pretrained -s de fr es -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Symmetry


# Inversion - Source 1
####################
# Target
#python run_reasoning.py -r INV_en_de-fr-es_target -s en -t de fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 500 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_de_en-fr-es_target -s de -t en fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_fr_de-en-es_target -s fr -t de en es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_es_de-en-fr_target -s es -t de en fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion

#python run_reasoning.py -r INV_en_de_target -s en -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_en_fr_target -s en -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_en_es_target -s en -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_de_en_target -s de -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_de_fr_target -s de -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_de_es_target -s de -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_fr_de_target -s fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_fr_en_target -s fr -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_fr_es_target -s fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_es_de_target -s es -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_es_en_target -s es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_es_fr_target -s es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion

# Source
#python run_reasoning.py -r INV_en_de-fr-es -s en -t de fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_de_en-fr-es -s de -t en fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_fr_de-en-es -s fr -t de en es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_es_de-en-fr -s es -t de en fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion

#python run_reasoning.py -r INV_en_de -s en -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_en_fr -s en -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_en_es -s en -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_de_en -s de -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_de_fr -s de -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_de_es -s de -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_fr_de -s fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_fr_en -s fr -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_fr_es -s fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_es_de -s es -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_es_en -s es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_es_fr -s es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion

# Source + Pretrained
#python run_reasoning.py -r INV_en_de-fr-es_pretrained -s en -t de fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_en-fr-es_pretrained -s de -t en fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_de-en-es_pretrained -s fr -t de en es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_de-en-fr_pretrained -s es -t de en fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion

#python run_reasoning.py -r INV_en_de_pretrained  -s en -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_fr_pretrained  -s en -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_es_pretrained  -s en -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_en_pretrained  -s de -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_fr_pretrained  -s de -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_es_pretrained  -s de -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_de_pretrained  -s fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_en_pretrained  -s fr -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_es_pretrained  -s fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_de_pretrained  -s es -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_en_pretrained  -s es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_fr_pretrained  -s es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion

# Target + Pretrained
#python run_reasoning.py -r INV_en_de-fr-es_target_pretrained -s en -t de fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 500 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_en-fr-es_target_pretrained -s de -t en fr es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_de-en-es_target_pretrained -s fr -t de en es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_de-en-fr_target_pretrained -s es -t de en fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion

#python run_reasoning.py -r INV_en_de_target_pretrained  -s en -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_fr_target_pretrained  -s en -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_es_target_pretrained  -s en -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_en_target_pretrained  -s de -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_fr_target_pretrained  -s de -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de_es_target_pretrained  -s de -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_de_target_pretrained  -s fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_en_target_pretrained  -s fr -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_fr_es_target_pretrained  -s fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_de_target_pretrained  -s es -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_en_target_pretrained  -s es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_es_fr_target_pretrained  -s es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion


# Inversion - Source 3
####################
# Source
#python run_reasoning.py -r INV_en-de-fr_es -s en de fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_en-de-es_fr -s en de es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_en-es-fr_de -s en es fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_de-fr-es_en -s de fr es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --group Inversion

# Source + Pretrained
#python run_reasoning.py -r INV_en-de-fr_es_pretrained -s en de fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en-de-es_fr_pretrained -s en de es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en-es-fr_de_pretrained -s en es fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de-fr-es_en_pretrained -s de fr es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 125 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion

# Target
#python run_reasoning.py -r INV_en-de-fr_es_target -s en de fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_en-de-es_fr_target -s en de es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_en-es-fr_de_target -s en es fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_de-fr-es_en_target -s de fr es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Inversion

# Target + Pretrained
#python run_reasoning.py -r INV_en-de-fr_es_target_pretrained -s en de fr -t es --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en-de-es_fr_target_pretrained -s en de es -t fr --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en-es-fr_de_target_pretrained -s en es fr -t de --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_de-fr-es_en_target_pretrained -s de fr es -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion


# Negation - Source 1
####################
# Target
#python run_reasoning.py -r NEG_en_de_target_20pair -s en -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_en_fr_target_20pair -s en -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_en_es_target_20pair -s en -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_de_en_target_20pair -s de -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_de_fr_target_20pair -s de -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_de_es_target_20pair -s de -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_fr_de_target_20pair -s fr -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_fr_en_target_20pair -s fr -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_fr_es_target_20pair -s fr -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_es_de_target_20pair -s es -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_es_en_target_20pair -s es -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_es_fr_target_20pair -s es -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation

#python run_reasoning.py -r NEG_en_en_20pair -s en -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_en_de_20pair -s en -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_de_de_20pair -s de -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_fr_fr_20pair -s fr -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_es_es_20pair -s es -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation


# Target + Pretrained
#python run_reasoning.py -r NEG_en_de-fr-es_target_20pair_pretrained -s en -t de fr es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_de_en-fr-es_target_20pair_pretrained -s de -t en fr es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_fr_de-en-es_target_20pair_pretrained -s fr -t de en es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_es_de-en-fr_target_20pair_pretrained -s es -t de en fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation

#python run_reasoning.py -r NEG_en_de_target_20pair_pretrained -s en -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_en_fr_target_20pair_pretrained -s en -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_en_es_target_20pair_pretrained -s en -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_de_en_target_20pair_pretrained -s de -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_de_fr_target_20pair_pretrained -s de -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_de_es_target_20pair_pretrained -s de -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_fr_de_target_20pair_pretrained -s fr -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_fr_en_target_20pair_pretrained -s fr -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_fr_es_target_20pair_pretrained -s fr -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_es_de_target_20pair_pretrained -s es -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_es_en_target_20pair_pretrained -s es -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation
#python run_reasoning.py -r NEG_es_fr_target_20pair_pretrained -s es -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --use_pretrained --group Negation

# Source
#python run_reasoning.py -r NEG_en_de-fr-es_10pair -s en -t de fr es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_de_en-fr-es_10pair -s de -t en fr es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_fr_de-en-es_10pair -s fr -t de en es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_es_de-en-fr_10pair -s es -t de en fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation

# Source + Pretrained
#python run_reasoning.py -r NEG_en_de-fr-es_10pair_pretrained -s en -t de fr es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_pretrained --group Negation
#python run_reasoning.py -r NEG_de_en-fr-es_10pair_pretrained -s de -t en fr es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_pretrained --group Negation
#python run_reasoning.py -r NEG_fr_de-en-es_10pair_pretrained -s fr -t de en es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_pretrained --group Negation
#python run_reasoning.py -r NEG_es_de-en-fr_10pair_pretrained -s es -t de en fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_pretrained --group Negation


# Negation - Source 3
####################
# Source
#python run_reasoning.py -r NEG_en-de-fr_es_20pair_f -s en de fr -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_en-de-es_fr_20pair_f -s en de es -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_en-es-fr_de_20pair_f -s en es fr -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_de-fr-es_en_20pair_f -s de fr es -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation

# Target
#python run_reasoning.py -r NEG_en-de-fr_es_target_10pair -s en de fr -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Negation
#python run_reasoning.py -r NEG_en-de-es_fr_target_10pair -s en de es -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Negation
#python run_reasoning.py -r NEG_en-es-fr_de_target_10pair -s en es fr -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Negation
#python run_reasoning.py -r NEG_de-fr-es_en_target_10pair -s de fr es -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Negation

#python run_reasoning.py -r NEG_en-de-fr_es_target_20pair -s en de fr -t es --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_en-de-es_fr_target_20pair -s en de es -t fr --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_en-es-fr_de_target_20pair -s en es fr -t de --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_de-fr-es_en_target_20pair -s de fr es -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation


# Implication - Source 1
####################
# Verify
#python run_reasoning.py -r IMP_en_en -s en -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r IMP_en_en_4000f -s en -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 4000 --n_pairs 20 --group Implication

# Kassner et al. Setup
#python run_reasoning.py -r IMP_en_en_kassner -s en -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 2 --precision_k 4 --group Implication
#python run_reasoning.py -r IMP_en_de-fr-es_kassner_10pair_p4 -s en -t de fr es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --precision_k 4 --group Implication

# Source
#python run_reasoning.py -r IMP_en_de-fr-es -s en -t de fr es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r IMP_de_en-fr-es -s de -t en fr es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r IMP_fr_de-en-es -s fr -t de en es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r IMP_es_de-en-fr -s es -t de en fr --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication

# Target
#python run_reasoning.py -r IMP_en_de-fr-es_4000f -s en -t de fr es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 4000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_de_en-fr-es_4000f -s de -t en fr es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 4000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_fr_de-en-es_4000f -s fr -t de en es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 4000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_es_de-en-fr_4000f -s es -t de en fr --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 4000 --n_pairs 20 --use_target --group Implication

# Implication - Source 3
####################
# Source
#python run_reasoning.py -r IMP_en-de-fr_es_20pair -s en de fr -t es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r IMP_en-de-es_fr_20pair -s en de es -t fr --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r IMP_en-es-fr_de_20pair -s en es fr -t de --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r IMP_de-fr-es_en_20pair -s de fr es -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication

#python run_reasoning.py -r IMP_en-de-fr_es_10pair -s en de fr -t es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Implication
#python run_reasoning.py -r IMP_en-de-es_fr_10pair -s en de es -t fr --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Implication
#python run_reasoning.py -r IMP_en-es-fr_de_10pair -s en es fr -t de --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Implication
#python run_reasoning.py -r IMP_de-fr-es_en_10pair -s de fr es -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Implication

# Target
#python run_reasoning.py -r IMP_en-de-fr_es_target_20pair -s en de fr -t es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_en-de-es_fr_target_20pair -s en de es -t fr --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_en-es-fr_de_target_20pair -s en es fr -t de --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_de-fr-es_en_target_20pair -s de fr es -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication

#python run_reasoning.py -r IMP_en-de-fr_es_target_10pair -s en de fr -t es --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Implication
#python run_reasoning.py -r IMP_en-de-es_fr_target_10pair -s en de es -t fr --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Implication
#python run_reasoning.py -r IMP_en-es-fr_de_target_10pair -s en es fr -t de --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Implication
#python run_reasoning.py -r IMP_de-fr-es_en_target_10pair -s de fr es -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --use_target --group Implication


# Composition - Source 1
####################
# Target
#python run_reasoning.py -r COMP_en_de_target -s en -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_en_fr_target -s en -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_en_es_target -s en -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_de_en_target_shorter -s de -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 120 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_de_en_target_lr4 -s de -t en --relation composition --lr 4e-5 --batch_size 256 --epochs 120 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_de_fr_target -s de -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_de_es_target_shorter -s de -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 120 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_de_es_target_lr4 -s de -t es --relation composition --lr 4e-5 --batch_size 256 --epochs 120 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_fr_de_target -s fr -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_fr_en_target -s fr -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_fr_es_target -s fr -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_es_de_target -s es -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_es_en_target -s es -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_es_fr_target -s es -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition

# Source
#python run_reasoning.py -r COMP_en_de-fr-es -s en -t de fr es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_de_en-fr-es -s de -t en fr es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_fr_de-en-es -s fr -t de en es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_es_de-en-fr -s es -t de en fr --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition

#python run_reasoning.py -r COMP_en_de_shorter -s en -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_en_de_lr4 -s en -t de --relation composition --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_en_fr -s en -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_en_es -s en -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_de_en -s de -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_de_fr -s de -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_de_es -s de -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_fr_de -s fr -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_fr_en -s fr -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_fr_es -s fr -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_es_de -s es -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_es_en -s es -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_es_fr -s es -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition

# Composition - Source 3
####################
# Target
#python run_reasoning.py -r COMP_en-de-fr_es_target -s en de fr -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_en-de-es_fr_target -s en de es -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_en-es-fr_de_target_shorter -s en es fr -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 80 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_en-es-fr_de_target_lr4 -s en es fr -t de --relation composition --lr 4e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_de-fr-es_en_target -s de fr es -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition

# Target + Enhanced
#python run_reasoning.py -r COMP_en-de-fr_es_target_enhanced2 -s en de fr -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --use_enhanced --group Composition
#python run_reasoning.py -r COMP_en-de-es_fr_target_enhanced2 -s en de es -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --use_enhanced --group Composition
#python run_reasoning.py -r COMP_en-es-fr_de_target_enhanced2 -s en es fr -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --use_enhanced --group Composition
#python run_reasoning.py -r COMP_de-fr-es_en_target_enhanced2 -s de fr es -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --use_enhanced --group Composition

# Source
#python run_reasoning.py -r COMP_en-de-fr_es -s en de fr -t es --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_en-de-es_fr -s en de es -t fr --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_en-es-fr_de -s en es fr -t de --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_de-fr-es_en -s de fr es -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition


# Source Performance for all rules in English
###################################
#python run_reasoning.py -r EQUI_en_en -s en -t en --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r SYM_en_en -s en -t en --relation symmetry --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r INV_en_en -s en -t en --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r NEG_en_en_20pair -s en -t en --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r IMP_en_en -s en -t en --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Implication
#python run_reasoning.py -r COMP_en_en -s en -t en --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
