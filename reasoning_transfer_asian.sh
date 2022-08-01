#!/bin/bash
# Equivalence
####################
# Source
#python run_reasoning.py -r EQUI_en_zh -s en -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_en_ja -s en -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_en_ru -s en -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_ru_ja -s ru -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_ru_zh -s ru -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_zh_ja -s zh -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_zh_ru -s zh -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_ja_zh -s ja -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence
#python run_reasoning.py -r EQUI_ja_ru -s ja -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --group Equivalence

# Target
#python run_reasoning.py -r EQUI_en_zh_target -s en -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_en_ja_target -s en -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_en_ru_target -s en -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_ru_ja_target -s ru -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_ru_zh_target -s ru -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_zh_ja_target -s zh -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_zh_ru_target -s zh -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_ja_zh_target -s ja -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence
#python run_reasoning.py -r EQUI_ja_ru_target -s ja -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --group Equivalence

# Source + Pretrained
#python run_reasoning.py -r EQUI_en_zh_pretrained -s en -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_ja_pretrained -s en -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_ru_pretrained -s en -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ru_ja_pretrained -s ru -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ru_zh_pretrained -s ru -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_zh_ja_pretrained -s zh -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_zh_ru_pretrained -s zh -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ja_zh_pretrained -s ja -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ja_ru_pretrained -s ja -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_pretrained --group Equivalence

# Target + Pretrained
#python run_reasoning.py -r EQUI_en_zh_target_pretrained -s en -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_ja_target_pretrained -s en -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_en_ru_target_pretrained -s en -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ru_ja_target_pretrained -s ru -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ru_zh_target_pretrained -s ru -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_zh_ja_target_pretrained -s zh -t ja --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_zh_ru_target_pretrained -s zh -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ja_zh_target_pretrained -s ja -t zh --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence
#python run_reasoning.py -r EQUI_ja_ru_target_pretrained -s ja -t ru --relation equivalence --lr 4e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Equivalence


# Symmetry
####################
# Source
#python run_reasoning.py -r SYM_en_zh -s en -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_en_ja -s en -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_en_ru -s en -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_ru_ja -s ru -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_ru_zh -s ru -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_zh_ja -s zh -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_zh_ru -s zh -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_ja_zh -s ja -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry
#python run_reasoning.py -r SYM_ja_ru -s ja -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Symmetry

# Target
#python run_reasoning.py -r SYM_en_zh_target -s en -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_en_ja_target -s en -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_en_ru_target -s en -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_ru_ja_target -s ru -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_ru_zh_target -s ru -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_zh_ja_target -s zh -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_zh_ru_target -s zh -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_ja_zh_target -s ja -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry
#python run_reasoning.py -r SYM_ja_ru_target -s ja -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Symmetry

# Source + Pretrained
#python run_reasoning.py -r SYM_en_zh_pretrained -s en -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_en_ja_pretrained -s en -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_en_ru_pretrained -s en -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_ru_ja_pretrained -s ru -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_ru_zh_pretrained -s ru -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_zh_ja_pretrained -s zh -t ja --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_zh_ru_pretrained -s zh -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_ja_zh_pretrained -s ja -t zh --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry
#python run_reasoning.py -r SYM_ja_ru_pretrained -s ja -t ru --relation symmetry --lr 4e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Symmetry


# Inversion
####################
# Source
#python run_reasoning.py -r INV_en_zh -s en -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_en_ja -s en -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_en_ru -s en -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_ru_ja -s ru -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_ru_zh -s ru -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_zh_ja -s zh -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_zh_ru -s zh -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_ja_zh -s ja -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion
#python run_reasoning.py -r INV_ja_ru -s ja -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --group Inversion

# Target
#python run_reasoning.py -r INV_en_zh_target -s en -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_en_ja_target -s en -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_en_ru_target -s en -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_ru_ja_target -s ru -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_ru_zh_target -s ru -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_zh_ja_target -s zh -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_zh_ru_target -s zh -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_ja_zh_target -s ja -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion
#python run_reasoning.py -r INV_ja_ru_target -s ja -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --group Inversion

# Source + Pretrained
#python run_reasoning.py -r INV_en_zh_pretrained -s en -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_ja_pretrained -s en -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_ru_pretrained -s en -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ru_ja_pretrained -s ru -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ru_zh_pretrained -s ru -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_zh_ja_pretrained -s zh -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_zh_ru_pretrained -s zh -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ja_zh_pretrained -s ja -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ja_ru_pretrained -s ja -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_pretrained --group Inversion

# Target
#python run_reasoning.py -r INV_en_zh_target_pretrained -s en -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_ja_target_pretrained -s en -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_en_ru_target_pretrained -s en -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ru_ja_target_pretrained -s ru -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ru_zh_target_pretrained -s ru -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_zh_ja_target_pretrained -s zh -t ja --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_zh_ru_target_pretrained -s zh -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ja_zh_target_pretrained -s ja -t zh --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion
#python run_reasoning.py -r INV_ja_ru_target_pretrained -s ja -t ru --relation inversion --lr 5e-5 --batch_size 256 --epochs 250 --n_relations 10 --n_facts 1000 --use_target --use_pretrained --group Inversion


# Negation
####################
# Source
#python run_reasoning.py -r NEG_en_zh_10pair -s en -t zh --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_en_ja_10pair -s en -t ja --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_en_ru_10pair -s en -t ru --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_ru_ja_10pair -s ru -t ja --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_ru_zh_10pair -s ru -t zh --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_zh_ja_20pair -s zh -t ja --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_zh_ru_10pair -s zh -t ru --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation
#python run_reasoning.py -r NEG_ja_zh_20pair -s ja -t zh --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --group Negation
#python run_reasoning.py -r NEG_ja_ru_10pair -s ja -t ru --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 10 --group Negation

# Target
#python run_reasoning.py -r NEG_en_zh_target_20pair -s en -t zh --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_en_ja_target_20pair -s en -t ja --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_en_ru_target_20pair -s en -t ru --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_ru_ja_target_20pair -s ru -t ja --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_ru_zh_target_20pair -s ru -t zh --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_zh_ja_target_20pair -s zh -t ja --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_zh_ru_target_20pair -s zh -t ru --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_ja_zh_target_20pair -s ja -t zh --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation
#python run_reasoning.py -r NEG_ja_ru_target_20pair -s ja -t ru --relation negation --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Negation


# Implication
####################
#python run_reasoning.py -r IMP_en_zh_target -s en -t zh --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_en_ja_target -s en -t ja --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_en_ru_target -s en -t ru --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_ru_ja_target -s ru -t ja --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_ru_zh_target -s ru -t zh --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_zh_ja_target -s zh -t ja --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_zh_ru_target -s zh -t ru --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_ja_zh_target -s ja -t zh --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication
#python run_reasoning.py -r IMP_ja_ru_target -s ja -t ru --relation implication --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 20 --use_target --group Implication


# Composition
####################
# Source
#python run_reasoning.py -r COMP_en_zh -s en -t zh --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_en_ja -s en -t ja --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_en_ru -s en -t ru --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_ru_ja -s ru -t ja --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_ru_zh -s ru -t zh --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_zh_ja -s zh -t ja --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_zh_ru -s zh -t ru --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_ja_zh -s ja -t zh --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition
#python run_reasoning.py -r COMP_ja_ru -s ja -t ru --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --group Composition

# Target
#python run_reasoning.py -r COMP_en_zh_target -s en -t zh --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_en_ja_target -s en -t ja --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_en_ru_target -s en -t ru --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_ru_ja_target -s ru -t ja --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_ru_zh_target -s ru -t zh --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_zh_ja_target -s zh -t ja --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_zh_ru_target -s zh -t ru --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_ja_zh_target -s ja -t zh --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition
#python run_reasoning.py -r COMP_ja_ru_target -s ja -t ru --relation composition --lr 5e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --n_pairs 100 --use_target --group Composition