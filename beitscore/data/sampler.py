import itertools
import math
import random
from collections import defaultdict

import torch


class ClusterBatchSampler(torch.utils.data.Sampler):
    """Batches drawn from one image cluster at a time, clusters visited in random order.
    Uses Python's `random`, seeded by scripts/train.py --seed."""
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size

        # Group indices by cluster and shuffle each
        self.cluster_to_indices = defaultdict(list)
        for idx, cluster in enumerate(dataset.clusters):
            self.cluster_to_indices[cluster].append(idx)

        self.cluster_ids = list(self.cluster_to_indices.keys())

    def __iter__(self):
        # Make a shuffled pool of indices per cluster (reset each epoch)
        pools = {
            c: iter(random.sample(idxs, len(idxs)))
            for c, idxs in self.cluster_to_indices.items()
        }
        active_clusters = set(self.cluster_ids)

        while active_clusters:
            # Pick a random cluster
            cluster_id = random.choice(list(active_clusters))
            pool = pools[cluster_id]

            # Try to pull a full batch from it
            batch = list(itertools.islice(pool, self.batch_size))

            if len(batch) < self.batch_size:
                # Cluster exhausted, remove it
                active_clusters.discard(cluster_id)
                if batch:
                    yield batch  # yield the last incomplete batch, or remove this line to drop it
            else:
                yield batch

    def __len__(self):
        # every cluster ends with its incomplete batch, if any (see __iter__)
        return sum(
            math.ceil(len(v) / self.batch_size)
            for v in self.cluster_to_indices.values()
        )
