import math
import random
import itertools

import numpy as np
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
from custom_trainer import logger


# Verifies that facts are not predicted by pretrained model
def verify_model_predict(facts):
    logger.info('Running model fact checking...')

    # Load Tokenizer and Model if not given
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

    queries = []
    entities2 = []

    for fact in facts:
        # Replace entity2 by [MASK]
        word_list = fact.split()
        entity2 = word_list[-1]
        query = fact.replace(entity2, '') + '[MASK]'

        queries.append(query)
        entities2.append(entity2)

    # Get Top 5 Tokens
    encoded_input = tokenizer(queries, return_tensors='pt', padding=True)
    token_logits = model(**encoded_input).logits

    mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]

    # Pick the [MASK] candidates with the highest logits.
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices.tolist()
    entities2_token = tokenizer(entities2, add_special_tokens=False)['input_ids']

    # Is Entity2 in Top 5 Tokens?
    for token, top5_token in zip(entities2_token, top_5_tokens):
        if token in top5_token:
            return True

    logger.info('No facts found!')
    return False


# Generate random pairs of numbers (indices into entity)
# Order doesn't matter, can't repeat
# i.e. ok is: (0,1), (1,2), (0,2) but not ok is (0,1),(1,0) or (0,0)
# Runs until exhausted or reached max_size
# possible to limit occurences of index
def gen_index_pairs(n, max_size=np.Inf, limit=np.Inf):
    pairs = set()
    ind = list()

    while len(pairs) < max_size:
        # return number between 0 and n (exclude)
        x, y = np.random.randint(n), np.random.randint(n)

        while ind.count(x) >= limit or ind.count(y) >= limit:
            x, y = np.random.randint(n), np.random.randint(n)

        i = 0
        while (x, y) in pairs or (y, x) in pairs or x == y:
            if i > 10:
                return
            x, y = np.random.randint(n), np.random.randint(n)
            i += 1

        ind.append(x)
        ind.append(y)

        pairs.add((x, y))
        yield x, y


# n: how many I have
# num_indices: how many I need
# generates max_size random unique indices (for indexing in what n is refering to)
def generate_unique_indices(n, num_indices):
    # if we can't generate unique indices because the data is too small
    if n < num_indices:
        # Generate indices with as few reusing as possible
        return generate_all_indices(n, num_indices)
    else:
        return generate_indices(n, num_indices, 1)


# Generates indices with as few reuing as possible
def generate_all_indices(n, num_indices):
    taken = []

    # Take all indices
    times = math.floor(num_indices / n)
    for i in range(times):
        taken += list(range(n))

    # Increase length by rest indices
    taken += list(range(num_indices - len(taken)))

    return taken


# Can be used to limit occurrence of subjects within a relation
def generate_indices(n, num_indices, reuse_count=1, used_indices=None, max_instance_excluded=np.Inf, last_indices=None):
    if used_indices is None:
        used_indices = []
    taken = []

    if last_indices is not None:
        # Reuse last_indices if not already used too much
        if all(used_indices.count(x) < max_instance_excluded for x in last_indices):
            return last_indices

    while len(taken) < num_indices:
        # return number between 0 and n (exclude)
        x = np.random.randint(n)

        i = 0
        # if x is already taken or excluded, I need to get another one
        while x in taken or used_indices.count(x) == max_instance_excluded:
            if i > n / 2:
                logger.warning(f'Index generation failed to get {num_indices} indices!')
                return
            x = np.random.randint(n)
            i += 1

        for _ in range(reuse_count):
            if len(taken) == num_indices:
                break
            taken.append(x)

    return taken


def generate_index_pairs(n, index_list, max_size=np.Inf):
    pairs = set()
    k = 0

    while len(pairs) < max_size:
        # return number between 0 and n (exclude)
        x = index_list[k]
        y = np.random.randint(n)

        i = 0
        while (x, y) in pairs or (y, x) in pairs or x == y:
            if i > 10:
                return
            y = np.random.randint(n)
            i += 1

        pairs.add((x, y))
        k += 1

        yield x, y


def generate_index_implication(n, index_list, max_size=np.Inf, sample_size=3):
    pairs = []
    k = 0

    # To adjust for f
    sample_size += 1

    # Create a list of the max_size rang
    pair_idx_list = [i for i in range(n) if i not in index_list]
    random.shuffle(pair_idx_list)  # for randomness

    # list has n_samples and we need max_size (I know, really bad naming..)
    n_samples = int(len(pair_idx_list) / sample_size)
    if n_samples < max_size:
        raise ValueError('Not enough samples than we need.')

    # Take max_size tuples of size sample_size
    while len(pairs) < max_size:
        indices = pair_idx_list[k * sample_size:(k + 1) * sample_size]

        pairs.append(tuple(indices))
        k += 1

    return pairs


def generate_index_implication_old(n, index_list, max_size=np.Inf, sample_size=3):
    rng = np.random.default_rng()
    pairs = []
    k = 0

    while len(pairs) < max_size:
        # return number between 0 and n (exclude)
        x = index_list[k]

        # Sample sample_size+1 indices (f + a,b,c..)
        indices = rng.choice(n, size=sample_size + 1, replace=False)

        i = 0
        while x in indices:
            if i > 1000:
                raise ValueError('Cant sample entities.')
            indices = rng.choice(n, size=sample_size + 1, replace=False)
            i += 1

        pairs.append(tuple(indices))
        k += 1

    return pairs


def generate_index_composition(n, index_list, max_size=np.Inf):
    rng = np.random.default_rng()
    pairs = []
    k = 0

    while len(pairs) < max_size:
        # return number between 0 and n (exclude)
        x = index_list[k]

        # Sample 2 indices
        f, g = rng.choice(n, size=2, replace=False)

        i = 0
        while x in (f, g) or (f, g) in pairs or (g, f) in pairs:
            if i > 1000:
                raise ValueError('Cant sample entities.')
            f, g = rng.choice(n, size=2, replace=False)
            i += 1

        pairs.append((f, g))
        k += 1

    return pairs


def generate_negation_triples(n, index_list, max_size=np.Inf, old_pairs=None):
    antonym_pairs = []
    k = 0

    if old_pairs is None:
        old_pairs = []

    while len(antonym_pairs) < max_size:
        # return number between 0 and n (exclude)
        x = index_list[k]
        y = np.random.randint(n)
        z = np.random.randint(n)

        i = 0
        while (y, z) in antonym_pairs or (z, y) in antonym_pairs or (y, z) in old_pairs or (z, y) in old_pairs or x == y or y == z or x == z:
            if i > 100:
                raise ValueError('Cant generate triples.')
            y = np.random.randint(n)
            z = np.random.randint(n)
            i += 1

        antonym_pairs.append((y, z))
        k += 1

    return antonym_pairs


def generate_index_triples(n, index_list, max_size=np.Inf, old_pairs=None):
    triples = []
    k = 0

    if old_pairs is None:
        old_pairs = []

    while len(triples) < max_size:
        # return number between 0 and n (exclude)
        x = index_list[k]
        y = np.random.randint(n)
        z = np.random.randint(n)

        i = 0
        while tuple_in_set((x, y, z), set(triples)) or (y, z) in old_pairs or (z, y) in old_pairs or x == y or y == z or x == z:
            if i > 100:
                raise ValueError('Cant generate triples.')
            y = np.random.randint(n)
            z = np.random.randint(n)
            i += 1

        triples.append((x, y, z))
        k += 1

    return triples


def tuple_in_set(idx_tuple, idx_set):
    perms = list(itertools.permutations(idx_tuple))
    for perm in perms:
        if perm in idx_set:
            return True
    return False


def contains_all(lst, elements):
    return all(x in lst for x in elements)
