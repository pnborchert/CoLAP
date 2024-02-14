import json, os
import numpy as np

# load tacred
tacred = json.load(open('dev.json'))

# sample sets of 250 samples per seed
seeds = [41,42,43,44,45]
samples = 250

sets = []
for seed in seeds:
    np.random.seed(seed)

    # sample uniformly from relation types
    relations = list(set([x['relation'] for x in tacred]))
    samples_per_relation = samples // len(relations)
    sampled = []
    for relation in relations:
        s_rel = [x for x in tacred if x['relation'] == relation]
        sampled.extend(np.random.choice(s_rel, samples_per_relation, replace=False))

    rel_rest = np.random.choice(relations, samples - len(sampled), replace=False)
    for relation in rel_rest:
        s_rel = [x for x in tacred if x['relation'] == relation]
        sampled.append(np.random.choice(s_rel, 1, replace=False)[0])
    sets.append(sampled)
   
assert all([len(x) == samples for x in sets])

# save to disk
for i, seed in enumerate(seeds):
    json.dump(sets[i], open(f'episode_en_{seed}.json', 'w'))