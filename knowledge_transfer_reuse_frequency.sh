#!/bin/bash
# Frequency Experiments
python run_knowledge.py -r KT_Freq_en_de-fr-es -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --evaluate_test
python run_knowledge.py -r KT_Freq_de_en-fr-es -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --evaluate_test
python run_knowledge.py -r KT_Freq_fr_de-en-es -s fr -t de en es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --evaluate_test
python run_knowledge.py -r KT_Freq_es_de-en-fr -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --evaluate_test

# Multilingual Frequency Experiments
python run_knowledge.py -r KTm_Freq_en_de-fr-es -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --multilingual --evaluate_test
python run_knowledge.py -r KTm_Freq_de_en-fr-es -s de -t en fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --multilingual --evaluate_test
python run_knowledge.py -r KTm_Freq_fr_de-en-es -s fr -t de en es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --multilingual --evaluate_test
python run_knowledge.py -r KTm_Freq_es_de-en-fr -s es -t de en fr --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --frequency_test --multilingual --evaluate_test

# Re-Use Experiments
# Subject within Relation
#python run_knowledge.py -r KT_Reuse_en_subject_1 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 1 --evaluate_test
#python run_knowledge.py -r KT_Reuse_en_subject_10 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 10 --evaluate_test
#python run_knowledge.py -r KT_Reuse_en_subject_100 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 100 --evaluate_test
#python run_knowledge.py -r KT_Reuse_en_subject_1000 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 1000 --evaluate_test

#python run_knowledge.py -r KTm_Reuse_en_subject_1 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 1 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_Reuse_en_subject_10 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 10 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_Reuse_en_subject_100 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 100 --multilingual --evaluate_test
#python run_knowledge.py -r KTm_Reuse_en_subject_1000 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_per_relation 1000 --multilingual --evaluate_test

# Subject across Relations
#python run_knowledge.py -r KT_Reuse_en_subject_all_1_5r -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 5 --n_facts 1000 --reuse_test --subject_all_relation 1 --evaluate_test
#python run_knowledge.py -r KT_Reuse_en_subject_all_1_500f -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 500 --reuse_test --subject_all_relation 1 --evaluate_test
#python run_knowledge.py -r KT_Reuse_en_subject_all_2 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_all_relation 2 --evaluate_test
#python run_knowledge.py -r KT_Reuse_en_subject_all_5 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_all_relation 5 --evaluate_test
#python run_knowledge.py -r KT_Reuse_en_subject_all_10 -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --subject_all_relation 10 --evaluate_test

# Use agnostic object but multilingual subject to properly control for the amount
#python run_knowledge.py -r KTm_Reuse_en_subject_all_1_5r_mObject -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 5 --n_facts 1000 --reuse_test --multilingual --subject_all_relation 1 --multilingual_object --evaluate_test
#python run_knowledge.py -r KTm_Reuse_en_subject_all_1_500f_mObject -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 500 --reuse_test --multilingual --subject_all_relation 1 --multilingual_object --evaluate_test
#python run_knowledge.py -r KTm_Reuse_en_subject_all_2_mObject -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --multilingual --subject_all_relation 2 --multilingual_object --evaluate_test
#python run_knowledge.py -r KTm_Reuse_en_subject_all_5_mObject -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --multilingual --subject_all_relation 5 --multilingual_object --evaluate_test
#python run_knowledge.py -r KTm_Reuse_en_subject_all_10_mObject -s en -t de fr es --lr 6e-5 --batch_size 256 --epochs 200 --n_relations 10 --n_facts 1000 --reuse_test --multilingual --subject_all_relation 10 --multilingual_object --evaluate_test
