"""Host-only synthetic aggregation comparison; no models, GPU or network."""
import gc
import json
import resource
import sys
from types import SimpleNamespace
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from master_script.core.aggregation import private_fedavg
from master_script.core.defenses import clipped_mean

mode = sys.argv[1]
initial = ndarrays_to_parameters([np.zeros(1_000_000, dtype=np.float32) for _ in range(3)])
results = [(None, SimpleNamespace(parameters=ndarrays_to_parameters(
    [np.full(1_000_000, i + 1., dtype=np.float32) for _ in range(3)]),
    num_examples=1, metrics={})) for i in range(4)]
gc.collect()
if mode == 'old':
    previous = parameters_to_ndarrays(initial)
    retained_initial = [p.copy() for p in previous]
    updates = [parameters_to_ndarrays(r.parameters) for _, r in results]
    averaged = clipped_mean(updates, previous, 1., 1., np.random.default_rng(7))
    for _, result in results:
        result.parameters = ndarrays_to_parameters(averaged)
    encoded, _ = FedAvg().aggregate_fit(1, results, [])
    capture = parameters_to_ndarrays(encoded)
    previous = parameters_to_ndarrays(encoded)
else:
    encoded = private_fedavg(results, initial, 1., 1., np.random.default_rng(7))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({'mode': mode, 'parameters': 3_000_000, 'clients': 4,
                  'peak_rss_mib': round(peak / (1024**2 if sys.platform == 'darwin' else 1024), 1)}))
