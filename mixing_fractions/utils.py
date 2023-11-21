import numpy as np
from numba import njit
from figaro.load import load_data
from pathlib import Path

@njit
def _logsumexp_jit(a, b):
    a_max = np.max(a)
    tmp = b * np.exp(a - a_max)
    return np.log(np.sum(tmp)) + a_max

def summary_files(fractions, assignments, event_names, model_names, out_folder = '.'):
    """
    Produce summary files about the run
    
    Arguments:
        np.ndarray samples:      samples produced by the run
        list-of-str event_names: names of the GW events
        list-of-str model_names: names of the models
    """
    out_folder = Path(out_folder)
    if not out_folder.exists():
        try:
            out_folder.mkdir()
        # Avoids issue with parallelisation
        except FileExistsError:
            pass
    # Events
    event_assignment = {event:{model: (assignments[i] == j).sum() for j, model in enumerate(model_names)} for i, event in enumerate(event_names)}
    counts           = np.array([[name]+list(dv.values()) for (name,dv) in event_assignment.items()])
    np.savetxt(Path(out_folder, 'summary_events.txt'), counts, fmt = '%s', header = 'event '+' '.join(model_names))
    # Models
    with open(Path(out_folder, 'summary_models.txt'), 'w') as f:
        for i, model in enumerate(event_names):
            low, median, high = np.percentile(fractions[:,i], [16, 50, 84])
            f.write('{0}: {1:.3f} + {2:.3f} - {3:.3f}'.format(model, median, high - median, median - low))
