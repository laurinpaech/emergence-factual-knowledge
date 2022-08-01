#!/bin/bash
# Agnostic - Few-Shot - 1 Source
#python run_knowledge.py -r KT_en_de_50shot -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_en_de_100shot -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_en_de_200shot -s en -t de --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot

#python run_knowledge.py -r KT_en_fr_50shot -s en -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_en_fr_100shot -s en -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_en_fr_200shot -s en -t fr --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot

#python run_knowledge.py -r KT_en_zh_50shot -s en -t zh --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_en_zh_100shot -s en -t zh --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_en_zh_200shot -s en -t zh --lr 6e-5 --batch_size 256 --epochs 180 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot

#python run_knowledge.py -r KT_ja_ru_50shot -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_ja_ru_100shot -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_ja_ru_200shot -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot

# DONT HAVE GOOD RESULTS:
#python run_knowledge.py -r KT_ja_ru_50shot_source -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --n_shot 50 --source_entities --multilingual --group n_shot
#python run_knowledge.py -r KT_ja_ru_100shot_source -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --n_shot 100 --source_entities --multilingual --group n_shot
#python run_knowledge.py -r KT_ja_ru_200shot_source -s ja -t ru --lr 6e-5 --batch_size 256 --epochs 150 --n_relations 10 --n_facts 1000 --n_shot 200 --source_entities --multilingual --group n_shot

# Agnostic - Few-Shot - 3 Source
#python run_knowledge.py -r KT_de-fr-es_en_50shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_de-fr-es_en_100shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_de-fr-es_en_200shot -s de fr es -t en --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot

#python run_knowledge.py -r KT_en-de-es_fr_50shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_en-de-es_fr_100shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_en-de-es_fr_200shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot

#python run_knowledge.py -r KT_en-fr-es_de_50shot -s en de es -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_en-de-es_de_100shot -s en de es -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_en-de-es_de_200shot -s en de es -t de --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot

#python run_knowledge.py -r KT_en-de-es_fr_50shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 50 --group n_shot
#python run_knowledge.py -r KT_en-de-es_fr_100shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 100 --group n_shot
#python run_knowledge.py -r KT_en-de-es_fr_200shot -s en de es -t fr --lr 6e-5 --batch_size 256 --epochs 100 --n_relations 10 --n_facts 1000 --n_shot 200 --group n_shot
