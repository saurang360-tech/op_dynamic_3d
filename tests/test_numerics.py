"""Behavioral contracts for the existing 2D opinion-dynamics model."""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from ODE_solver import social_network
from RDN import RDN
from runner_pre_RDN import save_simulation, run_single_simulation
from runner_post_RDN import save_rdn_output, run_rdn_simulation
from master import run_pre_rdn, run_post_rdn
from h5_loader import load_last_history_step, load_multiple
from Bulk_visualiser import load_folder


class MetricAndProbabilityTests(unittest.TestCase):
    def test_metric_distance_matches_phi_transform(self):
        sn = social_network(N=4, delta=50)
        first = np.array([0.75, -1.25], dtype=np.float32)
        second = np.array([-0.5, 0.25], dtype=np.float32)
        delta = first - second
        transformed = sn.phi @ delta
        expected = np.sqrt((transformed.T @ transformed).item())
        self.assertEqual(sn.dist(first, second), expected)

    def test_singular_metric_warns_but_right_angle_metric_does_not(self):
        with self.assertWarnsRegex(RuntimeWarning, 'singular or ill-conditioned'):
            social_network(N=3, delta=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            social_network(N=3, delta=90)
        self.assertEqual(caught, [])
        with self.assertRaisesRegex(ValueError, 'finite angle'):
            social_network(N=3, delta=float('nan'))

    def test_coincident_opinions_have_finite_normalized_probabilities(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sn = social_network(N=5, delta=0, beta=2)
        sn._opinions.fill(1.25)
        probabilities = sn.connect_prob_matrix(sn.beta)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_array_equal(np.diag(probabilities), np.zeros(5, dtype=np.float32))

    def test_probability_rows_normalize_and_diagonal_is_zero(self):
        sn = social_network(N=8, delta=50, beta=2)
        sn._opinions = np.array(
            [[i * 0.7, ((i * 3) % 7) * 0.4] for i in range(8)],
            dtype=np.float32,
        )
        probabilities = sn.connect_prob_matrix(sn.beta)
        np.testing.assert_array_equal(np.diag(probabilities), np.zeros(8))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_good_probability_rows_normalize_when_another_row_falls_back(self):
        sn = social_network(N=4, delta=50, beta=2)
        sn._opinions = np.array(
            [[0, 0], [1e20, 1e20], [0, 1], [2, 3]], dtype=np.float32
        )
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            probabilities = sn.connect_prob_matrix(sn.beta)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_array_equal(np.diag(probabilities), np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(
            probabilities[1], np.array([1 / 3, 0, 1 / 3, 1 / 3], dtype=np.float32)
        )

    def test_activity_inverse_cdf_uses_one_uniform_per_agent(self):
        sn = social_network(N=5, gamma=2.1, epsilon=0.01)
        uniforms = np.array([0.05, 0.2, 0.45, 0.7, 0.95], dtype=np.float32)
        with patch('ODE_solver.np.random.random', return_value=uniforms.copy()):
            sn.set_activity()
        first = sn.epsilon ** (1 - sn.gamma)
        expected = np.power(
            first + (1 - first) * uniforms, 1 / (1 - sn.gamma)
        ).astype(np.float32)
        np.testing.assert_array_equal(sn.activities, expected)


class DynamicsTests(unittest.TestCase):
    def test_adjacency_columns_are_active_sources(self):
        sn = social_network(N=4, m=1)
        sn.activities[:] = 1
        sn.p_conn = np.full((4, 4), 1 / 3, dtype=np.float32)
        np.fill_diagonal(sn.p_conn, 0)
        with patch('ODE_solver.np.random.random', return_value=np.zeros(4, dtype=np.float32)):
            with patch(
                'ODE_solver.np.random.choice',
                side_effect=[np.array([(agent + 1) % 4]) for agent in range(4)],
            ):
                adjacency = sn.create_adj_matrix().copy()
        expected = np.zeros((4, 4), dtype=np.int8)
        for source in range(4):
            expected[(source + 1) % 4, source] = 1
        np.testing.assert_array_equal(adjacency, expected)

    def test_adjacency_sparse_row_fallback_remains_uniform(self):
        sn = social_network(N=4, m=2)
        sn.activities[:] = 1
        sn.p_conn = np.zeros((4, 4), dtype=np.float32)
        observed = []

        def choose(population, size, replace, p):
            observed.append(np.asarray(p).copy())
            return np.array([0, 1])

        with patch('ODE_solver.np.random.random', return_value=np.zeros(4, dtype=np.float32)):
            with patch('ODE_solver.np.random.choice', side_effect=choose):
                sn.create_adj_matrix()
        self.assertEqual(len(observed), 4)
        for agent, probabilities in enumerate(observed):
            expected = np.full(4, 1 / 3, dtype=np.float32)
            expected[agent] = 0
            np.testing.assert_array_equal(probabilities, expected)

    def test_opinion_rhs_matches_documented_code_order(self):
        sn = social_network(N=3, K=1.7, alpha=1.2, alpha2=0.8, delta=40, RDN_select=0)
        opinions = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.25]], dtype=np.float32)
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int8)
        transformed = opinions @ sn.phi
        influence = np.column_stack((
            np.tanh(sn.alpha * transformed[:, 0]),
            np.tanh(sn.alpha2 * transformed[:, 1]),
        ))
        expected = -opinions + sn.K * (adjacency @ influence)
        sn._p_conn_dynamics_call_counter = 0
        sn._p_conn_refresh_every_dynamics_calls = 100
        with patch.object(sn, 'create_adj_matrix', return_value=adjacency):
            actual = sn.op_dynamics(0.0, opinions)
        np.testing.assert_array_equal(actual, expected)

    def test_rdn_unsupported_methods_fail_loudly(self):
        opinions = np.zeros((4, 2), dtype=np.float32)
        for method in (2, 3, 99):
            with self.subTest(method=method):
                with self.assertRaisesRegex(NotImplementedError, f"method {method}"):
                    RDN(method=method).calculate(opinions, 0.0)

    def test_rdn_off_method_remains_zero(self):
        opinions = np.ones((4, 2), dtype=np.float32)
        self.assertEqual(RDN(method=0).calculate(opinions, 1.0), 0.0)

    def test_rdn_seeded_sample_mean_formula(self):
        opinions = np.arange(24, dtype=np.float32).reshape(12, 2)
        mean = float(np.mean(opinions[:, 1]))
        np.random.seed(41)
        indices = np.random.choice(12, size=4, replace=False)
        expected = np.sqrt(4) * (np.mean(opinions[indices, 1]) - mean)
        np.random.seed(41)
        actual = RDN(method=1, n=4, topic=1).calculate(opinions, mean)
        self.assertEqual(actual, expected)

    def test_euler_records_pre_step_states_and_updates_final_state(self):
        sn = social_network(N=2)
        sn._opinions[:] = 0
        with patch.object(sn, 'op_dynamics', return_value=np.ones((2, 2), dtype=np.float32)):
            sn.run_network(dt=0.1, t_dur=0.2)
        self.assertEqual(sn.result.y.shape, (2, 2, 2))
        np.testing.assert_array_equal(sn.result.y[:, :, 0], np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_allclose(sn.result.y[:, :, 1], 0.1, atol=1e-7)
        np.testing.assert_allclose(sn.get_opinions(), 0.2, atol=1e-7)

    def test_probability_refresh_is_call_based_and_persists_between_runs(self):
        euler = social_network(N=3, m=1)
        euler.activities[:] = 0
        euler.set_social_interactions()
        with patch.object(euler, 'connect_prob_matrix', wraps=euler.connect_prob_matrix) as refresh:
            euler.run_network(dt=0.01, t_dur=0.09)
            self.assertEqual(refresh.call_count, 0)
            self.assertEqual(euler._p_conn_dynamics_call_counter, 9)
            euler.run_network(dt=0.01, t_dur=0.01)
            self.assertEqual(refresh.call_count, 1)
            self.assertEqual(euler._p_conn_dynamics_call_counter, 10)

        rk4 = social_network(N=3, m=1)
        rk4.activities[:] = 0
        rk4.set_social_interactions()
        with patch.object(rk4, 'connect_prob_matrix', wraps=rk4.connect_prob_matrix) as refresh:
            rk4.run_network_rk4(dt=0.01, t_dur=0.03)
            self.assertEqual(rk4._p_conn_dynamics_call_counter, 12)
            self.assertEqual(refresh.call_count, 1)
            rk4.run_network_rk4(dt=0.01, t_dur=0.02)
            self.assertEqual(rk4._p_conn_dynamics_call_counter, 20)
            self.assertEqual(refresh.call_count, 2)

    def test_rk4_builds_a_fresh_adjacency_at_each_stage(self):
        sn = social_network(N=3, m=1)
        sn.activities[:] = 0
        sn.set_social_interactions()
        with patch.object(sn, 'create_adj_matrix', wraps=sn.create_adj_matrix) as build:
            sn.run_network_rk4(dt=0.1, t_dur=0.2)
        self.assertEqual(build.call_count, 4 * len(sn.result.t))

    def test_rk4_calls_rhs_four_times_per_step(self):
        sn = social_network(N=2)
        sn._opinions[:] = 0
        with patch.object(sn, 'op_dynamics', return_value=np.ones((2, 2), dtype=np.float32)) as rhs:
            sn.run_network_rk4(dt=0.1, t_dur=0.2)
        self.assertEqual(rhs.call_count, 8)
        np.testing.assert_allclose(sn.get_opinions(), 0.2, atol=1e-7)


class OutputTests(unittest.TestCase):
    def test_standalone_pre_runner_returns_integrated_endpoint(self):
        params = {
            'N': 5, 'm': 1, 'K': 1.0, 'alpha': 1.0, 'alpha2': 1.0,
            'delta': 50.0, 'beta': 2.0, 'r': 0.5, 'gamma': 2.1,
            'epsilon': 0.01, 'RDN_select': 0,
        }
        history, _, endpoint = run_single_simulation(88, params, dt=0.01, t_dur=0.02)
        self.assertFalse(np.array_equal(endpoint, history[:, :, -1]))

    def test_standalone_post_runner_returns_integrated_endpoint(self):
        params = {
            'N': 5, 'm': 1, 'K': 1.0, 'alpha': 1.0, 'alpha2': 1.0,
            'delta': 50.0, 'beta': 2.0, 'r': 0.5, 'gamma': 2.1,
            'epsilon': 0.01, 'seed': 88,
        }
        initial = np.arange(10, dtype=np.float32).reshape(5, 2)
        history, _, endpoint = run_rdn_simulation(
            initial, params, dt=0.01, t_dur=0.02, D=0.0, sample_size=0,
            rdn_select=0, rdn_topic=0,
        )
        self.assertFalse(np.array_equal(endpoint, history[:, :, -1]))

    def test_pre_runner_hdf5_schema(self):
        history = np.zeros((3, 2, 4), dtype=np.float32)
        times = np.arange(4, dtype=np.float32)
        final = np.ones((3, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'run.h5'
            save_simulation(str(target), history, times, final, {'N': 3}, 17)
            with h5py.File(target, 'r') as handle:
                self.assertEqual(
                    set(handle.keys()),
                    {'opinions', 'time', 'final_opinions', 'last_history_step'},
                )
                self.assertEqual(handle['opinions'].shape, (3, 2, 4))
                self.assertEqual(handle['time'].shape, (4,))
                self.assertEqual(handle['final_opinions'].shape, (3, 2))
                np.testing.assert_array_equal(handle['final_opinions'][:], final)
                np.testing.assert_array_equal(handle['last_history_step'][:], history[:, :, -1])
                self.assertEqual(handle.attrs['seed'], 17)
                self.assertFalse(handle.attrs['rdn_applied'])

    def test_post_runner_writer_preserves_old_snapshot(self):
        history = np.zeros((3, 2, 4), dtype=np.float32)
        times = np.arange(4, dtype=np.float32)
        endpoint = np.full((3, 2), 2.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'post.h5'
            save_rdn_output(
                str(target), history, times, endpoint, {'N': 3},
                D=1.0, sample_size=2, rdn_select=1, rdn_topic=0,
            )
            with h5py.File(target, 'r') as handle:
                np.testing.assert_array_equal(handle['final_opinions'][:], endpoint)
                np.testing.assert_array_equal(handle['last_history_step'][:], history[:, :, -1])

    def test_master_pre_writer_saves_integrated_endpoint(self):
        params = {
            'N': 5, 'm': 1, 'K': 1.0, 'alpha': 1.0, 'alpha2': 1.0,
            'delta': 50.0, 'beta': 2.0, 'r': 0.5, 'gamma': 2.1,
            'epsilon': 0.01,
        }
        with tempfile.TemporaryDirectory() as folder:
            run_pre_rdn(params, dt=0.01, t_dur=0.02, seed=88, output_folder=folder)
            with h5py.File(Path(folder) / 'pre_seed_0088.h5', 'r') as handle:
                last = handle['opinions'][:, :, -1]
                np.testing.assert_array_equal(handle['last_history_step'][:], last)
                self.assertFalse(np.array_equal(handle['final_opinions'][:], last))

    def test_master_post_writer_saves_integrated_endpoint(self):
        params = {
            'N': 5, 'm': 1, 'K': 1.0, 'alpha': 1.0, 'alpha2': 1.0,
            'delta': 50.0, 'beta': 2.0, 'r': 0.5, 'gamma': 2.1,
            'epsilon': 0.01,
        }
        with tempfile.TemporaryDirectory() as folder:
            pre_folder = Path(folder) / 'pre'
            post_folder = Path(folder) / 'post'
            pre_folder.mkdir()
            with h5py.File(pre_folder / 'pre_seed_0088.h5', 'w') as handle:
                handle.create_dataset('final_opinions', data=np.arange(10, dtype=np.float32).reshape(5, 2))
                for key, value in params.items():
                    handle.attrs[key] = value
            run_post_rdn(
                str(pre_folder), params, dt=0.01, t_dur=0.02, seed=88,
                rdn_select=0, rdn_topic=0, rdn_D=0.0, rdn_n=0,
                output_folder=str(post_folder),
            )
            with h5py.File(post_folder / 'post_seed_0088.h5', 'r') as handle:
                last = handle['opinions'][:, :, -1]
                np.testing.assert_array_equal(handle['last_history_step'][:], last)
                self.assertFalse(np.array_equal(handle['final_opinions'][:], last))

    def test_loaders_and_visualizer_read_endpoint_and_legacy_snapshot(self):
        endpoint_new = np.full((2, 2), 7.0, dtype=np.float32)
        endpoint_old = np.full((2, 2), 9.0, dtype=np.float32)
        history_new = np.zeros((2, 2, 3), dtype=np.float32)
        history_old = np.ones((2, 2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            new_file = Path(folder) / 'new.h5'
            old_file = Path(folder) / 'old.h5'
            with h5py.File(new_file, 'w') as handle:
                handle.create_dataset('opinions', data=history_new)
                handle.create_dataset('final_opinions', data=endpoint_new)
                handle.create_dataset('last_history_step', data=history_new[:, :, -1])
            with h5py.File(old_file, 'w') as handle:
                handle.create_dataset('opinions', data=history_old)
                handle.create_dataset('final_opinions', data=endpoint_old)
            with tempfile.TemporaryDirectory() as loader_root:
                log_folder = Path(loader_root) / 'sim_data_log'
                log_folder.mkdir()
                (log_folder / 'new.h5').write_bytes(new_file.read_bytes())
                prior_cwd = os.getcwd()
                try:
                    os.chdir(loader_root)
                    np.testing.assert_array_equal(
                        load_last_history_step('new.h5'), history_new[:, :, -1]
                    )
                finally:
                    os.chdir(prior_cwd)
            final_data, count = load_folder(folder)
            self.assertEqual(count, 2)
            np.testing.assert_array_equal(np.sort(np.unique(final_data[:, 0])), [7.0, 9.0])
            last_history = load_multiple(folder, data_type='last_history_step')
            self.assertEqual(last_history.shape, (2, 2, 2))
            self.assertTrue(any(
                np.array_equal(sample, history_old[:, :, -1]) for sample in last_history
            ))

if __name__ == '__main__':
    unittest.main()
