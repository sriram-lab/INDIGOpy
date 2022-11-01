"""This module contains all custom functions at the core of the INDIGO algorithm."""

__licence__ = "GPL GNU"
__docformat__ = "reStructuredText"

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from warnings import warn
from itertools import compress


def load_sample(dataset:str): 
    """Loads a sample dataset. 
    
    This function loads a dictionary containing data relevant to a sample organism. 
    Currently supports data for *Escherichia coli*, *Mycobacterium tuberculosis*, 
    *Staphylococcus aureus*, and *Acinetobacter baumannii*.
    
    Parameters
    ----------
    dataset : str
        A string specifying the organism for which to load the sample data.
        Choose from 'ecoli', 'mtb', 'saureus', or 'abaumannii'. 
        
    Returns
    -------
    dict
        A dictionary object containing data for an organism of interest. Specifically:  
            * For 'ecoli', the dictionary contains the following keys: 
                * key: a dictionary for drug name mapping  
                * profiles: a dictionary of drug profile data (i.e., chemogenomic data)  
                * feature_names: a list of feature (i.e., gene) names associated with drug profile data  
                * train: a dictionary for the train subset of the drug interaction data  
                * test: a dictionary for the test subset of the drug interaction data  
            * For 'mtb', the dictionary contains the following keys:  
                * key: a dictionary for drug name mapping  
                * profiles: a dictionary of drug profile data (i.e., transcriptomic data)  
                * feature_names: a list of feature (i.e., gene) names associated with drug profile data  
                * train: a dictionary for the train subset of the drug interaction data  
                * test: a dictionary for the test subset of the drug interaction data  
                * clinical: a dictionary for the clinical subset of the drug interaction data 
            * For 'saureus', the dictionary contains the following keys: 
                * key: a dictionary for drug name mapping  
                * profiles: a dictionary of drug profile data (i.e., chemogenomic data)  
                * feature_names: a list of feature (i.e., gene) names associated with drug profile data  
                * train: a dictionary for the train subset of the drug interaction data  
                * test: a dictionary for the test subset of the drug interaction data 
                * orthology: a dictionary for the orthology data between *E. coli* and *S. aureus*  
            * For 'abaumannii', the dictionary contains the following keys: 
                * key: a dictionary for drug name mapping  
                * profiles: a dictionary of drug profile data (i.e., chemogenomic data)  
                * feature_names: a list of feature (i.e., gene) names associated with drug profile data  
                * train: a dictionary for the train subset of the drug interaction data  
                * test: a dictionary for the test subset of the drug interaction data 
                * orthology: a dictionary for the orthology data between *E. coli* and *A. baumannii*  
        
    Raises
    ------
    TypeError
        Raised when the input type is not a string.
    ValueError
        Raised when the function argument does not match accepted values ('ecoli', 'mtb', 'saureus', 'abaumannii'). 
    
    Examples
    --------
    Usage cases of the `load_sample` function.
    
    >>> ecoli_data = load_sample('ecoli')
    >>> print(ecoli_data['train']['interactions'][0])
    ['AMK', 'CEF']
    >>> mtb_data = load_sample('mtb')
    >>> print(mtb_data['clinical']['interactions'][0])
    ['EMBx', 'INH']
    >>> saureus_data = load_sample('saureus')
    >>> print(saureus_data['orthology']['map']['S_aureus'][0:3])
    ['b0002', 'b0003', 'b0007']
    >>> abaumannii_data = load_sample('abaumannii')
    >>> print(abaumannii_data['orthology']['map']['A_baumannii'][0:3])
    ['b0002', 'b0006', 'b0007']
    
    """
    # Check inputs
    if type(dataset) is not str: 
        raise TypeError('Provide a string input')
    if dataset not in ('ecoli', 'mtb', 'saureus', 'abaumannii'): 
        raise ValueError("Provide one of the following options: ('ecoli', 'mtb', 'saureus', 'abaumannii').")
    # Load queried dataset
    path = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(path, 'sample_data.pkl')
    with open(file, 'rb') as f: 
        ecoli_data, mtb_data, saureus_data, abaumannii_data = pickle.load(f)
        if dataset == 'ecoli': 
            return ecoli_data
        elif dataset == 'mtb': 
            return mtb_data
        elif dataset == 'saureus': 
            return saureus_data
        elif dataset == 'abaumannii': 
            return abaumannii_data


def featurize(interactions:list, profiles:dict, feature_names:list=None, key:list=None, 
              normalize:bool=False, norm_method:str='znorm', na_handle:float=0., 
              binarize:bool=True, thresholds:tuple=(-2, 2), remove_zero_rows:bool=False, 
              entropy:bool=False, time:bool=False, time_values:list=None, 
              strains:list=None, orthology_map:dict=None, silent:bool=False): 
    """Determines ML features for a list of drug combinations. 
    
    This function determines the feature information (i.e., joint profile) for a given drug combination. 
    The feature information is comprised of four pieces of information: 

        * **sigma scores**: indicative of the combined drug effect 
        * **delta scores**: indicative of drug-unique effects 
        * **entropy scores**: indicative of the combined entropy (optional) 
        * **time score**: indicative of time difference between treatments (optional) 
    
    Parameters
    ----------
    interactions : list
        A list of lists containing the drug names involved in a combination. 
    profiles : dict
        A dictionary of profile information for individual drug treatments. 
    feature_names : list, optional
        A list of feature names corresponding to the profile information (default is None). 
    key : list, optional
        A list of tuple pairs containing mapping information between drug names in interactions and profiles (default is None).
        The first element of each tuple must correspond to drug names in interactions.
        The second element of each tuple must exist in profiles.keys().
    normalize : bool, optional
        Boolean flag to normalize drug profile data (default is False). 
    norm_method : str, optional
        A string specifying the normalization method to use; choose between 'znorm' or 'minmax' (default is 'znorm'). 
    na_handle : float, optional
        A numeric value used for replacing any NaN values in profiles (default is 0.). 
    binarize : bool, optional
        Boolean flag to binarize drug profile data (default is True). 
    thresholds : tuple, optional
        A tuple of floating numbers indicative of (inclusive) thresholds for data binarization.
    remove_zero_rows : bool, optional
        Boolean flag for remove all-zero rows from profile data (default is False). 
    entropy : bool, optional
        Boolean flag to determine entropy scores (default is False). 
    time : bool, optional
        Boolean flag to determine time score (default is False). 
    time_values : list, optional
        A list of time values to use for the time feature (default is None). 
        Length must match length of interactions list.
    strains : list, optional
        A list of strain names that correspond to interactions (default is None). 
        Length must match length of interactions list.
    orthology_map : dict, optional
        A dictionary of orthology information for each unique strain in strains (default is None). 
        Key entries must match the unique strain names in strains.
    silent : bool, optional
        Boolean flag to silence warnings (default is False). 
        
    Returns
    -------
    dict
        A dictionary of feature information for the given list of drug combinations. 
        
    Raises
    ------
    AssertionError
        Raised when argument data dimensions do not correctly correspond to one another. 
    KeyError
        Raised when a drug profile is not provided; bypassed with a warning indicating missing information. 
    TypeError
        Raised when a given input type is incorrect. 
    ValueError
        Raised when a given input value is incorrect. 
    
    Examples
    --------
    Usage cases of the `featurize` function. 
    
    >>> interactions = [['A', 'B'], ['A', 'C'], ['B', 'C'], ['A', 'B', 'C']]
    >>> profiles = {'A': [1, 0, 1], 'B': [-2, 1.5, -0.5], 'C': [1, 2, 3]}
    >>> out = featurize(interactions, profiles)
    >>> print(out['feature_df'])
    
    .. csv-table::
        :header:  , A + B, A + C, B + C, A + B + C

        sigma-neg-feat1, 0, 0, 0, 0
        sigma-neg-feat2, 0, 0, 0, 0
        sigma-neg-feat3, 0, 0, 0, 0
        sigma-pos-feat1, 0, 0, 0, 0
        sigma-pos-feat2, 0, 0, 0, 0
        sigma-pos-feat3, 0, 1, 1, 0.666667
        delta-neg-feat1, 0, 0, 0, 0
        delta-neg-feat2, 0, 0, 0, 0
        delta-neg-feat3, 0, 0, 0, 0
        delta-pos-feat1, 0, 0, 0, 0
        delta-pos-feat2, 0, 0, 0, 0
        delta-pos-feat3, 0, 1, 1, 1

    >>> feature_names = ['G1', 'G2', 'G3']
    >>> out = featurize(interactions, profiles, feature_names=feature_names)
    >>> print(out['feature_df'])

    .. csv-table::
        :header:  , A + B, A + C, B + C, A + B + C

        sigma-neg-G1, 0, 0, 0, 0
        sigma-neg-G2, 0, 0, 0, 0
        sigma-neg-G3, 0, 0, 0, 0
        sigma-pos-G1, 0, 0, 0, 0
        sigma-pos-G2, 0, 0, 0, 0
        sigma-pos-G3, 0, 1, 1, 0.666667
        delta-neg-G1, 0, 0, 0, 0
        delta-neg-G2, 0, 0, 0, 0
        delta-neg-G3, 0, 0, 0, 0
        delta-pos-G1, 0, 0, 0, 0
        delta-pos-G2, 0, 0, 0, 0
        delta-pos-G3, 0, 1, 1, 1

    >>> profiles_alt = {'Drug_A': [1, 0, 1], 'Drug_B': [-2, 1.5, -0.5], 'Drug_C': [1, 2, 3]}
    >>> key = [('A', 'Drug_A'), ('B', 'Drug_B'), ('C', 'Drug_C')]
    >>> silent = True
    >>> out = featurize(interactions, profiles_alt, key=key, silent=silent)
    >>> print(out['feature_df'])

    .. csv-table::
        :header:  , A + B, A + C, B + C, A + B + C

        sigma-neg-feat1, 0, 0, 0, 0
        sigma-neg-feat2, 0, 0, 0, 0
        sigma-neg-feat3, 0, 0, 0, 0
        sigma-pos-feat1, 0, 0, 0, 0
        sigma-pos-feat2, 0, 0, 0, 0
        sigma-pos-feat3, 0, 1, 1, 0.666667
        delta-neg-feat1, 0, 0, 0, 0
        delta-neg-feat2, 0, 0, 0, 0
        delta-neg-feat3, 0, 0, 0, 0
        delta-pos-feat1, 0, 0, 0, 0
        delta-pos-feat2, 0, 0, 0, 0
        delta-pos-feat3, 0, 1, 1, 1

    >>> normalize, norm_method, na_handle = True, 'minmax', 0
    >>> out = featurize(interactions, profiles, normalize=normalize, norm_method=norm_method)
    >>> print(out['feature_df'])

    .. csv-table::
        :header:  , A + B, A + C, B + C, A + B + C

        sigma-neg-feat1, 0, 0, 0, 0
        sigma-neg-feat2, 0, 0, 0, 0
        sigma-neg-feat3, 0, 0, 0, 0
        sigma-pos-feat1, 0, 0, 0, 0
        sigma-pos-feat2, 0, 0, 0, 0
        sigma-pos-feat3, 0, 0, 0, 0
        delta-neg-feat1, 0, 0, 0, 0
        delta-neg-feat2, 0, 0, 0, 0
        delta-neg-feat3, 0, 0, 0, 0
        delta-pos-feat1, 0, 0, 0, 0
        delta-pos-feat2, 0, 0, 0, 0
        delta-pos-feat3, 0, 0, 0, 0

    >>> binarize, thresholds, remove_zero_rows = True, (-1, 1), True
    >>> out = featurize(interactions, profiles, binarize=binarize, thresholds=thresholds, remove_zero_rows=remove_zero_rows)
    >>> print(out['feature_df'])

    .. csv-table::
        :header:  , A + B, A + C, B + C, A + B + C

        sigma-neg-feat1, 1, 0, 1, 0.666667
        sigma-pos-feat2, 1, 1, 2, 1.333333
        sigma-pos-feat3, 0, 1, 1, 0.666667
        delta-neg-feat1, 1, 0, 1, 1
        delta-pos-feat2, 1, 1, 0, 0
        delta-pos-feat3, 0, 1, 1, 1

    >>> entropy, time, time_values = True, True, [[0, 0], [1, 1], [1, 2], [1, 2, 3]]
    >>> out = featurize(interactions, profiles, entropy=entropy, time=time, time_values=time_values)
    >>> print(out['feature_df'])

    .. csv-table::
        :header:  , A + B, A -> C, B -> C, A -> B -> C

        sigma-neg-feat1, 0, 0, 0, 0
        sigma-neg-feat2, 0, 0, 0, 0
        sigma-neg-feat3, 0, 0, 0, 0
        sigma-pos-feat1, 0, 0, 0, 0
        sigma-pos-feat2, 0, 0, 0, 0
        sigma-pos-feat3, 0, 1, 1, 0.666667
        delta-neg-feat1, 0, 0, 0, 0
        delta-neg-feat2, 0, 0, 0, 0
        delta-neg-feat3, 0, 0, 0, 0
        delta-pos-feat1, 0, 0, 0, 0
        delta-pos-feat2, 0, 0, 0, 0
        delta-pos-feat3, 0, 0.5, 0.666667, 0.5
        entropy-mean, 0.0136995, -0.549306, 0.563006, 0.00913299
        entropy-sum, 0.027399, -1.09861, 1.12601, 0.027399
        time, 0, 1, 1, 3

    >>> feature_names = ['G1', 'G2', 'G3']
    >>> strains = ['MG1655', 'MG1655', 'MC1400', 'IAI1']
    >>> orthology_map = {'MG1655': ['G1', 'G2'], 'MC1400': ['G1', 'G3'], 'IAI1': ['G1']}
    >>> out = featurize(interactions, profiles, feature_names=feature_names, strains=strains, orthology_map=orthology_map)
    >>> print(out['feature_df'])

    .. csv-table::
        :header:  , A + B, A + C, B + C, A + B + C

        sigma-neg-G1, 0, 0, 0, 0
        sigma-neg-G2, 0, 0, 0, 0
        sigma-neg-G3, 0, 0, 0, 0
        sigma-pos-G1, 0, 0, 0, 0
        sigma-pos-G2, 0, 0, 0, 0
        sigma-pos-G3, 0, 0, 1, 0
        delta-neg-G1, 0, 0, 0, 0
        delta-neg-G2, 0, 0, 0, 0
        delta-neg-G3, 0, 0, 0, 0
        delta-pos-G1, 0, 0, 0, 0
        delta-pos-G2, 0, 0, 0, 0
        delta-pos-G3, 0, 1, 1, 1

    """
    # Check inputs
    if any([type(x) is not bool for x in (normalize, binarize, remove_zero_rows, entropy, time, silent)]): 
        raise AssertionError('Invalid value (not Boolean) provided for normalize, binarize, remove_zero_rows, entropy, time, or silent')
    if type(interactions) is not list or not all([type(x) is list for x in interactions]): 
        raise TypeError('Provide a list of list elements for interactions')
    drug_list = set([drug for interaction in interactions for drug in interaction])
    if not all([type(x) is str for x in drug_list]): 
        raise TypeError('Entries for each interaction must be of str type')
    if type(profiles) is not dict or not any([drug in drug_list for drug in profiles.keys()]): 
        if key is None or not any(drug in [entry[1] for entry in key] for drug in profiles.keys()):
            raise TypeError('Provide a dict with valid key entries for profiles')
        else: 
            if silent is False: 
                warn('Interaction entries do not match profile names, but key provided')
    profile_values = [entry for value in profiles.values() for entry in value]
    n = int(len(profile_values) / len(profiles.keys()))
    if any([type(x) is not list for x in profiles.values()]) or any([len(x) != n for x in profiles.values()]): 
        raise AssertionError('Ensure that all values in profiles are list objects of equal size')
    if not all([type(x) is int or type(x) is float for x in profile_values]): 
        raise TypeError('Ensure that all elements in profile values are real numeric types')
    if feature_names is not None: 
        if type(feature_names) is not list or any([type(x) is not str for x in feature_names]): 
            raise TypeError('Provide a list of strings for feature_names')
        if len(feature_names) != n: 
            raise AssertionError('Provide {} elements for feature_names'.format(n))
    else: 
        feature_names = ['feat{}'.format(i) for i in range(1, n+1)]
    if key is not None: 
        if (type(key) is not list or not any([drug in [entry[1] for entry in key] for drug in profiles.keys()]) or 
            not any([drug in [entry[0] for entry in key] for drug in drug_list])): 
                raise TypeError('Provide a list with valid tuple entries for key')
    if normalize and norm_method not in ('znorm', 'minmax'): 
        raise ValueError('Provide "znorm" or "minmax" for norm_method')
    if not any([type(na_handle) is int or type(na_handle) is float]): 
        raise TypeError('Provide a real numeric value for na_handle')
    if type(thresholds) is not tuple or not all([type(x) is int or type(x) is float for x in thresholds]): 
        raise TypeError('Provide a tuple of real numeric elements for thresholds')
    if len(thresholds) != 2: 
        raise AssertionError('Provide 2 elements for thresholds argument')
    if time_values is not None: 
        time = True
        if type(time_values) is not list or not all([type(x) is list for x in interactions]): 
            raise TypeError('Provide a list of list elements for time_values')
        time_entries = [entry for t in time_values for entry in t]
        if (not all([type(x) is int or type(x) is float for x in time_entries]) and 
            not all([x >= 0 for x in time_entries])):
                raise ValueError('All time value entries must be whole numbers')
        if len(time_values) != len(interactions): 
            raise AssertionError('Length of time_values must match length of interactions')
    elif time_values is None and time: 
        time_values = []
        for ixn in interactions: 
            time_values.append([0] * len(ixn))
    if strains is not None: 
        if not all([type(x) is str for x in strains]): 
            raise TypeError('All entries in strains must be of str type')
        if len(strains) != len(interactions): 
            raise AssertionError('Length of strains must match length of interactions (N = {})'.format(len(interactions)))
        if orthology_map is None: 
            raise ValueError('orthology_map is None while strains is not None. Provide a dict for orthology_map')
        else: 
            if (type(orthology_map) is not dict or not all([strain in orthology_map.keys() for strain in strains])): 
                    raise TypeError('Provide a dict with valid key entries for orthology_map')

    # Modify profile names (if key is provided)
    df = pd.DataFrame.from_dict(profiles)
    if key is not None: 
        # df = pd.concat([df, df.rename(columns=key)], axis=1)
        for entry in key: 
            df[entry[0]] = df[entry[1]]
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Extract relevant drug profiles
    keep_ixns = [all(i in list(df.columns) for i in ixn) for ixn in interactions]
    ixn_list = [ixn for ixn in interactions if all(i in list(df.columns) for i in ixn)]
    if len(ixn_list) < len(interactions): 
        if silent is False: 
            warn('Drug profile information missing for {} interactions'.format(len(interactions) - len(ixn_list)))
    df = df[list(set([drug for ixn in ixn_list for drug in ixn]))]

    # Handle NaNs
    df = df.fillna(na_handle)

    # Apply data processing (as prompted)
    if normalize: 
        if norm_method == 'znorm': 
            df = (df - df.mean()) / df.std()
        elif norm_method == 'minmax': 
            df = (df - df.min()) / (df.max() - df.min())
    if binarize: 
        bin_df = pd.concat([df < thresholds[0], df > thresholds[1]], ignore_index=True)
        feature_names = ['neg-' + feature for feature in feature_names] + ['pos-' + feature for feature in feature_names]
    else: 
        bin_df = df
    if remove_zero_rows: 
        zero_index = (bin_df==0).all(axis=1)
        bin_df = bin_df.loc[~zero_index]
        feature_names = list(compress(feature_names, (~zero_index).tolist()))

    # Determine ML features
    feature_list = ['sigma-' + feature for feature in feature_names] + ['delta-' + feature for feature in feature_names]
    if entropy: 
        feature_list = feature_list + ['entropy-mean', 'entropy-sum']
    if time: 
        feature_list.append('time')
    feature_dict = {}
    for i, ixn in enumerate(tqdm(ixn_list, desc='Defining INDIGO features')): 
        sigma = bin_df[ixn].sum(axis=1) * (2 / len(ixn))
        if time is False or (time is True and time_values is None): 
            delta = (bin_df[ixn].sum(axis=1) == 1).astype('float')
            delim = ' + '
        elif time is True and time_values is not None: 
            if all(t == 0 for t in time_values[i]): 
                delta = (bin_df[ixn].sum(axis=1) == 1).astype('float')
                delim = ' + '
            else: 
                delta = pd.Series(np.diff((bin_df[ixn] * time_values[i]).values, n=len(ixn)-1).flatten() / sum(time_values[i]))
                delim = ' -> '
        feat_data = sigma.tolist() + delta.tolist()
        if entropy: 
            feat_data = feat_data + [np.log(df[ixn].var()).mean(), np.log(df[ixn].var()).sum()]
        if time: 
            feat_data = feat_data + [sum(time_values[i][:-1])]
        feature_dict_keys = [key.split(' - dup')[0] for key in list(feature_dict.keys())]
        n_key = feature_dict_keys.count(delim.join(ixn))
        if n_key >= 1: 
            feature_dict[delim.join(ixn) + ' - dup' + str(n_key)] = feat_data
        else: 
            feature_dict[delim.join(ixn)] = feat_data
    feature_df = pd.DataFrame.from_dict(feature_dict)
    feature_df.index = feature_list

    # Apply orthology mapping (if strains is not None)
    if strains is not None: 
        strains = list(compress(strains, keep_ixns))
        strain_set = set(strains)
        for strain in tqdm(strain_set, desc='Mapping orthologous genes'): 
            orthologs = orthology_map[strain]
            row_mask = [feature for feature in list(feature_df.index) if (feature.startswith('sigma')) & (not any([gene == feature[10:] for gene in orthologs]))]
            col_mask = [s == strain for s in strains]
            feature_df.loc[row_mask, col_mask] = 0

    return {'interaction_list': ixn_list, 'drug_profiles': df, 'feature_df': feature_df, 'idx': keep_ixns}


def classify(scores:list, thresholds:tuple=(-0.1, 0.1), classes:tuple=('Synergy', 'Neutral', 'Antagonism')): 
    """Converts drug interaction scores into interaction classes. 
    
    Score-to-class conversion is based on the general convention for synergy (negative) and antagonism (positive)
    for interaction outcomes measured based on the Loewe Additivity or Bliss Independence models. 
    
    Parameters
    ----------
    scores : list
        A list of drug interaction scores. 
    thresholds : tuple, optional
        A tuple of two floating numbers indicative of (inclusive) thresholds for synergy and antagonism, repectively.
    classes : tuple, optional
        A tuple of three strings representative of class labels. By default, the three classes are Synergy, Neutral, and Antagonism.
        
    Returns
    -------
    list
        A list of class labels for the given list of interaction scores. 
        
    Raises
    ------
    AssertionError
        Raised when the wrong number of elements is provided for `thresholds` or `classes`. 
    TypeError
        Raised when a given input type is incorrect. 
    
    Examples
    --------
    Usage cases of the `classify` function. 
    
    >>> scores = [-2, 1.5, 0.5, -0.1, 1]
    >>> classify(scores)
    ['Synergy', 'Antagonism', 'Antagonism', 'Synergy', 'Antagonism']
    >>> thresholds = (-1, 1)
    >>> classify(scores, thresholds=thresholds)
    ['Synergy', 'Antagonism', 'Neutral', 'Neutral', 'Antagonism']
    >>> classes = ('S', 'N', 'A')
    >>> classify(scores, thresholds=thresholds, classes=classes)
    ['S', 'A', 'N', 'N', 'A']
    
    """
    # Check inputs
    if type(scores) is not list or not all([type(x) is int or type(x) is float for x in scores]): 
        raise TypeError('Provide a list of real numeric elements for scores')
    if type(thresholds) is not tuple or not all([type(x) is int or type(x) is float for x in thresholds]): 
        raise TypeError('Provide a tuple of real numeric elements for thresholds')
    if len(thresholds) != 2: 
        raise AssertionError('Provide 2 elements for thresholds argument')
    if type(classes) is not tuple or any([type(x) is not str for x in classes]): 
        raise TypeError('Provide a tuple of string elements for classes')
    if len(classes) != 3: 
        raise AssertionError('Provide 3 elements for classes argument')

    # Determine class type for scores
    labels = []
    for x in scores: 
        if x <= thresholds[0]: 
            labels.append(classes[0])
        elif x >= thresholds[1]: 
            labels.append(classes[2])
        else: 
            labels.append(classes[1])

    return labels

