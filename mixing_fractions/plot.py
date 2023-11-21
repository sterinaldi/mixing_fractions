import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from figaro import plot_settings
from pathlib import Path
from corner import corner

# Telling python to ignore empty legend warning from matplotlib
warnings.filterwarnings("ignore", message = "No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.")

def single_model_histogram(samples, name, out_folder = '.'):
    """
    Plot histogram for the mixing fraction of a single model
    
    Arguments:
        np.ndarray samples: mixing fraction samples for the model
        str name:           name of the model
        str out_folder:     output folder
    """
    out_folder = Path(out_folder)
    fig, ax = plt.subplots()
    ax.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$p(w)$')
    fig.align_labels()
    fig.savefig(Path(out_folder, name + '.pdf'), bbox_inches = 'tight')

def single_event_histogram(samples, name, model_names = None, out_folder = '.'):
    """
    Plot histogram for the assignment of a single event
    
    Arguments:
        np.ndarray samples:      mixing fraction samples for the model
        list-of-str model_names: model names
        str out_folder:          output folder
        str out_folder:          output folder
    """
    out_folder = Path(out_folder)
    fig, ax = plt.subplots()
    if model_names is not None:
        if not (len(model_names) == samples.shape[-1]):
            raise IndexError('The number of model names do not match the samples shape')
        if plot_settings.tex_flag:
            model_names = ['$\\mathrm{'+name+'}$' for name in model_names]
            ax.set_xlabel('$\\mathrm{Model}$')
            ax.set_ylabel('$p(\\mathrm{Model})$')
        else:
            model_names = ['$\mathrm{'+name+'}$' for name in model_names]
            ax.set_xlabel('$\mathrm{Model}$')
            ax.set_ylabel('$p(\mathrm{Model})$')
        dict_vals = {name: (samples == i).sum() for i, name in enumerate(model_names)}
    else:
        model_names = [None for _ in range(samples.shape[-1])]
    instances = np.array([(samples == i).sum() for i, name in enumerate(model_names)])
    ax.stairs(values = instances/len(samples), edges = np.arange(0, len(model_names)+1), color = 'steelblue')
    ax.stairs(values = (instances + np.sqrt(instances))/len(samples), baseline = (instances - np.sqrt(instances))/len(samples), fill = True, edges = np.arange(0, len(model_names)+1), color = 'steelblue', alpha = 0.25)
    ax.set_xticks(np.arange(0, len(model_names)) + 0.5)
    ax.set_xticklabels(model_names)
    ax.tick_params(axis = 'x', bottom = False, labelrotation = 45)
    fig.savefig(Path(out_folder, name + '.pdf'), bbox_inches = 'tight')

def joint_posterior_histogram(samples, model_names = None, out_folder = '.', colormap = 'jet'):
    """
    Plot histogram for the mixing fraction of a single model
    
    Arguments:
        np.ndarray samples:      mixing fraction samples
        list-of-str model_names: model names
        str out_folder:          output folder
        str colormap:            colormap
    """
    out_folder = Path(out_folder)
    if model_names is not None:
        if not (len(model_names) == samples.shape[-1]):
            raise IndexError('The number of model names do not match the samples shape')
        if plot_settings.tex_flag:
            model_names = ['$\\mathrm{'+name+'}$' for name in model_names]
        else:
            model_names = ['$\mathrm{'+name+'}$' for name in model_names]
    else:
        model_names = [None for _ in range(samples.shape[-1])]
    # Corner plot
    fig = corner(samples, color = '#1f77b4', fig = fig, hist_kwargs = {'density': True, 'linewidth':0.7} , plot_density = False, contour_kwargs = {'linewidths':0.3, 'linestyles':'dashed'}, levels = [0.5,0.68,0.9], no_fill_contours = True, hist_bin_factor = int(np.sqrt(len(samples)))/20., quiet = True, labels = model_names)
    fig.savefig(Path(out_folder, 'joint_posterior.pdf'), bbox_inches = 'tight')
    # Histograms
    color = iter(colormaps[colormap](np.linspace(0, 1, samples.shape[-1])))
    fig, ax = plt.subplots()
    for s, name in zip(samples.T, model_names):
        c = next(color)
        ax.hist(s, bins = int(np.sqrt(len(s))), histtype = 'step', color = 'c', density = True, label = name)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$p(w)$')
    ax.legend(loc = 0)
    fig.align_labels()
    fig.savefig(Path(out_folder, 'joint_histogram.pdf'), bbox_inches = 'tight')
