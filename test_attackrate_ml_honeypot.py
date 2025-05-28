import unittest
import pandas as pd
import os
from attackrate_ml_honeypot_refactored import (
    split_xyz, normalize_attack_labels, add_difference_features,
    identify_benign_senders, analyze_top_senders,
    train_and_evaluate_first_attack_model, load_dataframes_from_csv
)

class TestHoneypotFunctions(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'sender': [1, 2, 3, 1, 2, 3],
            'Attack': [0, 1, 0, 22, 33, 0],
            'pos_x': [0, 1, 2, 3, 4, 5],
            'pos_y': [0, -1, -2, -3, -4, -5],
            'vel_x': [1, 1, 1, 1, 1, 1],
            'vel_y': [0, 0, 0, 0, 0, 0]
        })

    def test_normalize_attack_labels(self):
        result = normalize_attack_labels(self.df.copy())
        self.assertTrue(all(result['Attack'].isin([0, 22, 33])))

    def test_add_difference_features(self):
        result = add_difference_features(self.df.copy(), ['pos_x', 'pos_y'])
        self.assertIn('diff_pos_x', result.columns)
        self.assertIn('diff_pos_y', result.columns)

    def test_identify_benign_senders(self):
        benign_df = identify_benign_senders(self.df.copy(), attack_label=1)
        self.assertIsInstance(benign_df, pd.DataFrame)
        self.assertIn('sender', benign_df.columns)

    def test_analyze_top_senders(self):
        benign_df = identify_benign_senders(self.df.copy(), attack_label=1)
        top_df = analyze_top_senders(benign_df, top_n=2)
        self.assertLessEqual(len(top_df), 2)

class TestModelAndLoading(unittest.TestCase):

    def setUp(self):
        # Create temporary CSV for loading tests
        self.df = pd.DataFrame({
            'pos_x': [0, 1, 2, 3, 4, 5],
            'pos_y': [0, 1, 2, 3, 4, 5],
            'vel_x': [1, 1, 1, 1, 1, 1],
            'vel_y': [0, 0, 0, 0, 0, 0],
            'Attack': [0, 22, 0, 33, 0, 0]
        })

        self.test_dir = "./test_data_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.csv_file = os.path.join(self.test_dir, "sample.csv")
        self.df.to_csv(self.csv_file, index=False)

    def tearDown(self):
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_train_and_evaluate_first_attack_model(self):
        result = train_and_evaluate_first_attack_model(self.df.copy())
        self.assertIn('rf_model', result)
        self.assertIn('gb_model', result)
        self.assertIn('X_test', result)
        self.assertEqual(len(result['X_test']), len(result['rf_predictions']))

    def test_load_dataframes_from_csv(self):
        regular_dfs, ground_truth_dfs = load_dataframes_from_csv(self.test_dir)
        self.assertTrue(len(regular_dfs) == 1)
        self.assertTrue(isinstance(regular_dfs[0], pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
