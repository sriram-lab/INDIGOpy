"""This script executes testing of core INDIGOpy functions."""


import numpy as np
import pandas as pd
from warnings import catch_warnings

from indigopy.core import load_sample, featurize, classify


class TestLoadSample: 

    input1 = 'ecoli'
    input2 = 'mtb'
    input3 = 'saureus'
    input4 = 'abaumannii'

    def test_load_sample_ecoli(self): 
        """Make sure that `load_sample` works for E. coli data."""
        sample = load_sample(self.input1)
        assert all(key in sample.keys() for key in ['key', 'profiles', 'feature_names', 'train', 'test'])
        assert all(key in sample['train'].keys() for key in ['interactions', 'scores'])
        assert all(key in sample['test'].keys() for key in ['interactions', 'scores'])
        assert sample['train']['interactions'][0] == ['AMK', 'CEF']
        out = featurize(sample['test']['interactions'], sample['profiles'], silent=True, 
                        feature_names=sample['feature_names'], key=sample['key'])
        assert all(key in out.keys() for key in ['interaction_list', 'drug_profiles', 'feature_df', 'idx'])

    def test_load_sample_mtb(self): 
        """Make sure that `load_sample` works for M. tuberculosis data."""
        sample = load_sample(self.input2)
        assert all(key in sample.keys() for key in ['key', 'profiles', 'feature_names', 'train', 'test', 'clinical'])
        assert all(key in sample['train'].keys() for key in ['interactions', 'scores'])
        assert all(key in sample['test'].keys() for key in ['interactions', 'scores'])
        assert all(key in sample['clinical'].keys() for key in ['interactions', 'scores'])
        assert sample['clinical']['interactions'][0] == ['EMBx', 'INH']
        out = featurize(sample['test']['interactions'], sample['profiles'], silent=True, 
                        feature_names=sample['feature_names'], key=sample['key'])
        assert all(key in out.keys() for key in ['interaction_list', 'drug_profiles', 'feature_df', 'idx'])

    def test_load_sample_saureus(self): 
        """Make sure that `load_sample` works for S. aureus data."""
        sample = load_sample(self.input3)
        assert all(key in sample.keys() for key in ['key', 'profiles', 'feature_names', 'train', 'test', 'orthology'])
        assert all(key in sample['train'].keys() for key in ['interactions', 'scores'])
        assert all(key in sample['test'].keys() for key in ['interactions', 'scores'])
        assert all(key in sample['orthology'].keys() for key in ['strains', 'map'])
        assert sample['orthology']['map']['S_aureus'][0:3] == ['b0002', 'b0003', 'b0007']
        out = featurize(sample['test']['interactions'], sample['profiles'], 
                        feature_names=sample['feature_names'], key=sample['key'], silent=True, 
                        strains=sample['orthology']['strains'], orthology_map=sample['orthology']['map'])
        assert all(key in out.keys() for key in ['interaction_list', 'drug_profiles', 'feature_df', 'idx'])

    def test_load_sample_abaumannii(self): 
        """Make sure that `load_sample` works for A. baumannii data."""
        sample = load_sample(self.input4)
        assert all(key in sample.keys() for key in ['key', 'profiles', 'feature_names', 'train', 'test', 'orthology'])
        assert all(key in sample['train'].keys() for key in ['interactions', 'scores'])
        assert all(key in sample['test'].keys() for key in ['interactions', 'scores'])
        assert all(key in sample['orthology'].keys() for key in ['strains', 'map'])
        assert sample['orthology']['map']['A_baumannii'][0:3] == ['b0002', 'b0006', 'b0007']
        out = featurize(sample['test']['interactions'], sample['profiles'], silent=True, 
                        feature_names=sample['feature_names'], key=sample['key'], 
                        strains=sample['orthology']['strains'], orthology_map=sample['orthology']['map'])
        assert all(key in out.keys() for key in ['interaction_list', 'drug_profiles', 'feature_df', 'idx'])


class TestFeaturize: 

    interactions        = [['A', 'B'], ['A', 'C'], ['B', 'C'], ['A', 'B', 'C']]
    profiles            = {'A': [1, float('nan'), 1], 'B': [-2, 1.5, -0.5], 'C': [1, 2, 3]}
    profiles_alt        = {'Drug_A': [1, float('nan'), 1], 'Drug_B': [-2, 1.5, -0.5], 'Drug_C': [1, 2, 3]}
    feature_names       = ['G1', 'G2', 'G3']
    key                 = [('A', 'Drug_A'), ('B', 'Drug_B'), ('C', 'Drug_C')]
    normalize           = True
    norm_method         = 'minmax'
    na_handle           = 0
    binarize            = False
    thresholds          = (-1, 1)
    remove_zero_rows    = True
    entropy             = True
    time                = True
    time_values         = [[0, 0], [1, 1], [1, 2], [1, 2, 3]]
    strains             = ['MG1655', 'MG1655', 'MC1400', 'IAI1']
    orthology_map       = {'MG1655': ['G1', 'G2'], 'MC1400': ['G1', 'G3'], 'IAI1': ['G1']}
    silent              = False
    
    def test_featurize_default(self): 
        """Make sure that `featurize` works with required inputs only."""
        out = featurize(self.interactions, self.profiles)
        df = pd.DataFrame(
            {
                'A + B': [0.0] * 12, 
                'A + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                'B + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                'A + B + C': [0.0] * 5 + [2/3] + [0.0] * 5 + [1.0]
            }
        )
        row_names = [
            'sigma-neg-feat1', 'sigma-neg-feat2', 'sigma-neg-feat3', 
            'sigma-pos-feat1', 'sigma-pos-feat2', 'sigma-pos-feat3', 
            'delta-neg-feat1', 'delta-neg-feat2', 'delta-neg-feat3', 
            'delta-pos-feat1', 'delta-pos-feat2', 'delta-pos-feat3'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_feature_names(self): 
        """Make sure that `featurize` works with the *feature_names* parameter."""
        out = featurize(self.interactions, self.profiles, feature_names=self.feature_names)
        df = pd.DataFrame(
            {
                'A + B': [0.0] * 12, 
                'A + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                'B + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                'A + B + C': [0.0] * 5 + [2/3] + [0.0] * 5 + [1.0]
            }
        )
        row_names = [
            'sigma-neg-G1', 'sigma-neg-G2', 'sigma-neg-G3', 
            'sigma-pos-G1', 'sigma-pos-G2', 'sigma-pos-G3', 
            'delta-neg-G1', 'delta-neg-G2', 'delta-neg-G3', 
            'delta-pos-G1', 'delta-pos-G2', 'delta-pos-G3'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_key_and_silent(self): 
        """Make sure that `featurize` works with the *key* parameter provided mismatching *interactions* and *profiles*. 
        Also check that a warning is raised. """
        with catch_warnings(record=True) as w: 
            out = featurize(self.interactions, self.profiles_alt, key=self.key, silent=self.silent)
            df = pd.DataFrame(
                {
                    'A + B': [0.0] * 12, 
                    'A + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                    'B + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                    'A + B + C': [0.0] * 5 + [2/3] + [0.0] * 5 + [1.0]
                }
            )
            row_names = [
                'sigma-neg-feat1', 'sigma-neg-feat2', 'sigma-neg-feat3', 
                'sigma-pos-feat1', 'sigma-pos-feat2', 'sigma-pos-feat3', 
                'delta-neg-feat1', 'delta-neg-feat2', 'delta-neg-feat3', 
                'delta-pos-feat1', 'delta-pos-feat2', 'delta-pos-feat3'
                ]
            df.index = row_names
            pd.testing.assert_frame_equal(out['feature_df'], df)
            assert len(w) > 0
            assert all(s in str(w[0].message) for s in ['profile', 'key'])

    def test_featurize_normalize_and_norm_method(self): 
        """Make sure that `featurize` works with *normalize* and *norm_method* parameters."""
        out = featurize(self.interactions, self.profiles, normalize=self.normalize, norm_method=self.norm_method)
        df = pd.DataFrame(
            {
                'A + B': [0.0] * 12, 
                'A + C': [0.0] * 12, 
                'B + C': [0.0] * 12, 
                'A + B + C': [0.0] * 12
            }
        )
        row_names = [
            'sigma-neg-feat1', 'sigma-neg-feat2', 'sigma-neg-feat3', 
            'sigma-pos-feat1', 'sigma-pos-feat2', 'sigma-pos-feat3', 
            'delta-neg-feat1', 'delta-neg-feat2', 'delta-neg-feat3', 
            'delta-pos-feat1', 'delta-pos-feat2', 'delta-pos-feat3'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_na_handle(self): 
        """Make sure that `featurize` works with the *na_handle* parameter."""
        out = featurize(self.interactions, self.profiles, na_handle=self.na_handle)
        df = pd.DataFrame(
            {
                'A + B': [0.0] * 12, 
                'A + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                'B + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                'A + B + C': [0.0] * 5 + [2/3] + [0.0] * 5 + [1.0]
            }
        )
        row_names = [
            'sigma-neg-feat1', 'sigma-neg-feat2', 'sigma-neg-feat3', 
            'sigma-pos-feat1', 'sigma-pos-feat2', 'sigma-pos-feat3', 
            'delta-neg-feat1', 'delta-neg-feat2', 'delta-neg-feat3', 
            'delta-pos-feat1', 'delta-pos-feat2', 'delta-pos-feat3'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_binarize(self): 
        """Make sure that `featurize` works with the *binarize* parameter."""
        out = featurize(self.interactions, self.profiles, binarize=self.binarize)
        df = pd.DataFrame(
            {
                'A + B': [-1.0, 1.5, 0.5] + [0.0] * 3, 
                'A + C': [2.0, 2.0, 4.0] + [0.0] * 3, 
                'B + C': [-1.0, 3.5, 2.5] + [0.0] * 3, 
                'A + B + C': [0.0, 7/3, 7/3] + [0.0] * 3
            }
        )
        row_names = [
            'sigma-feat1', 'sigma-feat2', 'sigma-feat3', 
            'delta-feat1', 'delta-feat2', 'delta-feat3'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_thresholds(self): 
        """Make sure that `featurize` works with the *thresholds* parameter."""
        out = featurize(self.interactions, self.profiles, thresholds=self.thresholds)
        df = pd.DataFrame(
            {
                'A + B': [1.0] + [0.0] * 3 + [1.0, 0.0, 1.0] + [0.0] * 3 + [1.0, 0.0], 
                'A + C': [0.0] * 4 + [1.0, 1.0] + [0.0] * 4 + [1.0, 1.0], 
                'B + C': [1.0] + [0.0] * 3 + [2.0, 1.0, 1.0] + [0.0] * 4 + [1.0], 
                'A + B + C': [2/3] + [0.0] * 3 + [4/3, 2/3, 1.0] + [0.0] * 4 + [1.0]
            }
        )
        row_names = [
            'sigma-neg-feat1', 'sigma-neg-feat2', 'sigma-neg-feat3', 
            'sigma-pos-feat1', 'sigma-pos-feat2', 'sigma-pos-feat3', 
            'delta-neg-feat1', 'delta-neg-feat2', 'delta-neg-feat3', 
            'delta-pos-feat1', 'delta-pos-feat2', 'delta-pos-feat3'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_remove_zero_rows(self): 
        """Make sure that `featurize` works with the *remove_zero_rows* parameter."""
        out = featurize(self.interactions, self.profiles, remove_zero_rows=self.remove_zero_rows)
        df = pd.DataFrame(
            {
                'A + B': [0.0, 0.0], 
                'A + C': [1.0, 1.0], 
                'B + C': [1.0, 1.0], 
                'A + B + C': [2/3, 1.0]
            }
        )
        row_names = ['sigma-pos-feat3', 'delta-pos-feat3']
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_entropy(self): 
        """Make sure that `featurize` works with the *entropy* parameter."""
        out = featurize(self.interactions, self.profiles, entropy=self.entropy)
        pdf = pd.DataFrame.from_dict(self.profiles)
        pdf = pdf.fillna(0.)
        entropy_AB = [np.log(pdf[['A', 'B']].var()).mean(), np.log(pdf[['A', 'B']].var()).sum()]
        entropy_AC = [np.log(pdf[['A', 'C']].var()).mean(), np.log(pdf[['A', 'C']].var()).sum()]
        entropy_BC = [np.log(pdf[['B', 'C']].var()).mean(), np.log(pdf[['B', 'C']].var()).sum()]
        entropy_ABC = [np.log(pdf[['A', 'B', 'C']].var()).mean(), np.log(pdf[['A', 'B', 'C']].var()).sum()]
        df = pd.DataFrame(
            {
                'A + B': [0.0] * 12 + entropy_AB, 
                'A + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0] + entropy_AC, 
                'B + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0] + entropy_BC, 
                'A + B + C': [0.0] * 5 + [2/3] + [0.0] * 5 + [1.0] + entropy_ABC
            }
        )
        row_names = [
            'sigma-neg-feat1', 'sigma-neg-feat2', 'sigma-neg-feat3', 
            'sigma-pos-feat1', 'sigma-pos-feat2', 'sigma-pos-feat3', 
            'delta-neg-feat1', 'delta-neg-feat2', 'delta-neg-feat3', 
            'delta-pos-feat1', 'delta-pos-feat2', 'delta-pos-feat3', 
            'entropy-mean', 'entropy-sum'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_time_and_time_values(self): 
        """Make sure that `featurize` works with *time* and *time_values* parameters."""
        out = featurize(self.interactions, self.profiles, time=self.time, time_values=self.time_values)
        time_AB = [sum(self.time_values[0][:-1])]
        time_AC = [sum(self.time_values[1][:-1])]
        time_BC = [sum(self.time_values[2][:-1])]
        time_ABC = [sum(self.time_values[3][:-1])]
        df = pd.DataFrame(
            {
                'A + B': [0.0] * 12 + time_AB, 
                'A -> C': [0.0] * 5 + [1.0] + [0.0] * 5 + [0.5] + time_AC, 
                'B -> C': [0.0] * 5 + [1.0] + [0.0] * 5 + [2/3] + time_BC, 
                'A -> B -> C': [0.0] * 5 + [2/3] + [0.0] * 5 + [0.5] + time_ABC
            }
        )
        row_names = [
            'sigma-neg-feat1', 'sigma-neg-feat2', 'sigma-neg-feat3', 
            'sigma-pos-feat1', 'sigma-pos-feat2', 'sigma-pos-feat3', 
            'delta-neg-feat1', 'delta-neg-feat2', 'delta-neg-feat3', 
            'delta-pos-feat1', 'delta-pos-feat2', 'delta-pos-feat3', 
            'time'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)

    def test_featurize_strains_and_orthology_map(self): 
        """Make sure that `featurize` works with *strains* and *orthology_map* parameters."""
        out = featurize(self.interactions, self.profiles, feature_names=self.feature_names, strains=self.strains, orthology_map=self.orthology_map)
        df = pd.DataFrame(
            {
                'A + B': [0.0] * 12, 
                'A + C': [0.0] * 11 + [1.0], 
                'B + C': [0.0] * 5 + [1.0] + [0.0] * 5 + [1.0], 
                'A + B + C': [0.0] * 11 + [1.0]
            }
        )
        row_names = [
            'sigma-neg-G1', 'sigma-neg-G2', 'sigma-neg-G3', 
            'sigma-pos-G1', 'sigma-pos-G2', 'sigma-pos-G3', 
            'delta-neg-G1', 'delta-neg-G2', 'delta-neg-G3', 
            'delta-pos-G1', 'delta-pos-G2', 'delta-pos-G3'
            ]
        df.index = row_names
        pd.testing.assert_frame_equal(out['feature_df'], df)


class TestClassify: 

    scores      = [-2, 1.5, 0.5, -0.1, 1]
    thresholds  = (-1, 1)
    classes     = ('S', 'N', 'A')

    def test_classify_default(self): 
        """Make sure that `classify` works with required inputs only."""
        assert classify(self.scores) == ['Synergy', 'Antagonism', 'Antagonism', 'Synergy', 'Antagonism']

    def test_classify_thresholds(self): 
        """Make sure that `classify` works with the *thresholds* parameter."""
        assert classify(self.scores, thresholds=self.thresholds) == ['Synergy', 'Antagonism', 'Neutral', 'Neutral', 'Antagonism']

    def test_classify_classes(self): 
        """Make sure that `classify` works with the *classes* parameter."""
        assert classify(self.scores, classes=self.classes) == ['S', 'A', 'A', 'S', 'A']

