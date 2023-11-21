import numpy as np

from tqdm import tqdm
from scipy.stats import dirichlet

from mixing_fractions.montecarlo import MC_integral
from mixing_fractions.utils import logsumexp_jit

class Gibbs:
    """
    Class to infer the mixing fractions of a set of formation channels using GW observations.
    
    Arguments:
        iterable posterior_samples: GW posterior samples
        iterable models:            formation channels models. Must be an iterable of callables
        iterable event_names:       GW event names
        iterable model_names:       formation channels names
        double alpha0:              concentration parameter
        int thinning:               number of steps between draws
    
    Returns:
        Gibbs: instance of Gibbs class
    """
    def __init__(self,
                 posterior_samples,
                 models,
                 event_names,
                 model_names,
                 alpha0 = 1.,
                 thinning = 1000,
                 ):
                 
        self.posterior_samples = posterior_samples
        self.models            = models
        self.event_names       = event_names
        self.model_names       = model_names
        self.n_events          = len(posterior_samples)
        self.n_models          = len(models)
        # Sampling setup
        self.alpha0            = alpha0
        self.alpha             = self.alpha0/self.n_models
        self.thinning          = int(thinning)
        # Initialisation
        self._evaluate_event_probabilities()
        self._initialise_assignments()

    def _evaluate_event_probabilities(self):
        """
        Evaluate the probability for each event of being generated by each model
        """
        self.event_probabilities = np.zeros((len(self.posterior_samples), len(self.models)))
        for i, event in tqdm(enumerate(self.posterior_samples), total = len(self.posterior_samples), desc = 'Evaluating probabilities'):
            self.event_probabilities[i] = np.array([MC_integral(model, event) for model in self.models])

    def _draw_assignment(self, i):
        """
        Draw a category assignment for event i
        
        Arguments:
            int i: index of the event
        
        Returns:
            idx: index of the selected component
        """
        # Compute probability for categories
        logP_DD    = np.log((self.counts + self.alpha)/(i + self.alpha0)) # Dirichlet Distribution
        logP       = logP_DD + self.event_probabilities - logsumexp_jit(self.event_probabilities[i], logP_DD)
        # Draw assignment
        idx        = np.random.choice(self.n_models, p = np.exp(logP))
        return idx

    def _initialise_assignments(self):
        """
        Initialise the assignments in a way that ensures immediate thermalisation
        """
        self.assignments = [None for _ in self.posterior_samples]
        self.counts      = np.zeros(len(self.models))
        order            = np.arange(self.n_events)
        np.random.shuffle(order)
        for i in order:
            idx = self._draw_assignment(i)
            self.assignments[i] = idx
            self.counts[idx]   += 1.
        
    def _update_component(self, i):
        """
        Update the component to which event i is assigned to

        Arguments:
            int i: index of the event
        """
        # Remove event from old component
        old_idx               = self.assignments[i]
        self.counts[old_idx] -= 1.
        # Draw new component
        new_idx               = self._draw_assignment(i)
        self.assignments[i]   = new_idx
        self.counts[new_idx] += 1.
    
    def _draw_mixing_fractions(self):
        """
        Draw a single mixing fractions realisation from a Dirichlet distribution conditioned on the current assignments
        
        Returns:
            fractions: mixing fractions
        """
        fractions = dirichlet((counts + self.alpha)/(self.n_events + self.alpha0)).rvs().flatten()
        return fractions
    
    def _draw_sample(self):
        """
        Draw an uncorrelated sample for the mixing fractions
        """
        event_indexes = np.random.choice(self.n_events, size = self.thinning, replace = True)
        for idx in event_indexes:
            self._update_component(idx)
        return self._draw_mixing_fractions()
    
    def rvs(self, n_draws = 1)
        """
        Draw random samples from the distribution
        
        Arguments:
            int n_draws: number of draws
        
        Returns:
            samples: samples for the mixing fractions
        """
        return np.array([self._draw_sample() for _ in tqdm(range(int(n_draws)), desc = 'Sampling', disable = (n_draws > 1))])