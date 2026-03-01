"""Tests for MatchFlow._internal.labeler module.

This module provides Labeler abstractions and implementations.
"""
import pytest
from unittest.mock import patch
import time

from MatchFlow._internal.labeler import (
    Labeler,
    GoldLabeler,
    DelayedGoldLabeler,
    CLILabeler,
    CustomLabeler,
    WebUILabeler,
)


class TestLabelerBase:
    """Tests for Labeler abstract interface."""

    def test_requires_implementation(self):
        """Assert abstract methods raise when not implemented."""
        with pytest.raises(TypeError):
            Labeler()

        class IncompleteLabeler(Labeler):
            pass

        with pytest.raises(TypeError):
            IncompleteLabeler()

        class CompleteLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        labeler = CompleteLabeler()
        assert labeler(1, 2) == 1.0


class TestGoldLabeler:
    """Tests for GoldLabeler behavior."""

    def test_gold_labeler_returns_labels(self, spark_session):
        """Verify GoldLabeler returns provided gold labels."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 11, "id2": 21}, {"id1": 12, "id2": 22}])
        labeler = GoldLabeler(gold_df)

        assert labeler(10, 20) == 1.0
        assert labeler(11, 21) == 1.0
        assert labeler(12, 22) == 1.0

        assert labeler(10, 21) == 0.0
        assert labeler(13, 23) == 0.0
        assert labeler(99, 99) == 0.0


class TestDelayedGoldLabeler:
    """Tests for DelayedGoldLabeler behavior."""

    def test_delayed_gold_labeler(self, spark_session):
        """Ensure DelayedGoldLabeler defers labels until available."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 11, "id2": 21}])
        delay_secs = 0.01
        labeler = DelayedGoldLabeler(gold_df, delay_secs)

        start_time = time.time()
        result = labeler(10, 20)
        elapsed = time.time() - start_time

        assert result == 1.0
        assert elapsed >= delay_secs

        start_time = time.time()
        result = labeler(10, 21)
        elapsed = time.time() - start_time

        assert result == 0.0
        assert elapsed >= delay_secs

    def test_delayed_gold_labeler_zero_delay(self, spark_session):
        """Test DelayedGoldLabeler with zero delay."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}])
        labeler = DelayedGoldLabeler(gold_df, 0.0)

        result = labeler(10, 20)
        assert result == 1.0

        result = labeler(10, 21)
        assert result == 0.0


class TestCLILabeler:
    """Tests for CLILabeler interactions."""

    def test_cli_labeler_init(self, a_df, b_df):
        """Test CLILabeler initialization."""
        labeler = CLILabeler(a_df, b_df, id_col='_id')

        assert labeler._a_df is a_df
        assert labeler._b_df is b_df
        assert labeler._id_col == '_id'
        assert labeler._all_fields is None
        assert labeler._current_fields is None

    def test_cli_labeler_get_row(self, a_df, b_df):
        """Test CLILabeler _get_row method."""
        labeler = CLILabeler(a_df, b_df)

        row = labeler._get_row(a_df, 10)
        assert isinstance(row, dict)
        assert row['_id'] == 10
        assert 'a_attr' in row
        assert 'a_num' in row

        with pytest.raises(KeyError, match='No row'):
            labeler._get_row(a_df, 999)

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_yes(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler with 'yes' input."""
        mock_input.return_value = 'y'
        labeler = CLILabeler(a_df, b_df)

        result = labeler(10, 20)

        assert result == 1.0
        assert mock_input.called

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_no(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler with 'no' input."""
        mock_input.return_value = 'n'
        labeler = CLILabeler(a_df, b_df)

        result = labeler(10, 20)

        assert result == 0.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_unsure(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler with 'unsure' input."""
        mock_input.return_value = 'u'
        labeler = CLILabeler(a_df, b_df)

        result = labeler(10, 20)

        assert result == 2.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_stop(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler with 'stop' input."""
        mock_input.side_effect = ['s', 'y']
        labeler = CLILabeler(a_df, b_df)

        result = labeler(10, 20)

        assert result == -1.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_stop_cancel(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler with 'stop' then cancel."""
        mock_input.side_effect = ['s', 'n', 'y']
        labeler = CLILabeler(a_df, b_df)

        result = labeler(10, 20)

        assert result == 1.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_help(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler with 'help' input."""
        mock_input.side_effect = ['h', 'e', 'y']
        labeler = CLILabeler(a_df, b_df)

        result = labeler(10, 20)

        assert result == 1.0
        assert mock_print.called

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_invalid_then_valid(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler with invalid input then valid."""
        mock_input.side_effect = ['invalid', 'x', 'y']
        labeler = CLILabeler(a_df, b_df)

        result = labeler(10, 20)

        assert result == 1.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_field_initialization(
        self, mock_print, mock_input, spark_session, a_df, b_df
    ):
        """Test CLILabeler field initialization."""
        mock_input.return_value = 'y'
        labeler = CLILabeler(a_df, b_df)

        assert labeler._all_fields is None
        assert labeler._current_fields is None

        labeler(10, 20)

        assert labeler._all_fields is not None
        assert isinstance(labeler._all_fields, list)
        assert labeler._current_fields is not None
        assert isinstance(labeler._current_fields, set)


class TestCustomLabeler:
    """Tests for CustomLabeler behavior."""

    def test_custom_labeler_requires_implementation(self):
        """Test that CustomLabeler is abstract and requires label_pair."""
        with pytest.raises(TypeError):
            CustomLabeler(None, None)

    def test_custom_labeler_callable(self, spark_session, a_df, b_df):
        """Ensure CustomLabeler delegates to provided callable."""
        class MyCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                if row1.get('a_attr') == row2.get('a_attr'):
                    return 1.0
                return 0.0

        labeler = MyCustomLabeler(a_df, b_df)

        result = labeler(10, 20)
        assert result == 1.0

        result = labeler(10, 21)
        assert result == 0.0

    def test_custom_labeler_get_row(self, spark_session, a_df, b_df):
        """Test CustomLabeler _get_row method."""
        class MyCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                return 1.0

        labeler = MyCustomLabeler(a_df, b_df)

        row = labeler._get_row(a_df, 10)
        assert isinstance(row, dict)
        assert row['_id'] == 10

        with pytest.raises(KeyError, match='No row'):
            labeler._get_row(a_df, 999)

    def test_custom_labeler_with_complex_logic(
        self, spark_session, a_df, b_df
    ):
        """Test CustomLabeler with more complex labeling logic."""
        class ThresholdLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                diff = abs(row1.get('a_num', 0) - row2.get('a_num', 0))
                return 1.0 if diff < 0.5 else 0.0

        labeler = ThresholdLabeler(a_df, b_df)

        result = labeler(10, 20)
        assert result == 1.0

        result = labeler(10, 21)
        assert result == 0.0


class TestWebUILabeler:
    """Tests for WebUILabeler behavior."""

    def test_webui_labeler_init(self, spark_session, a_df, b_df):
        """Test WebUILabeler initialization."""
        labeler = WebUILabeler(
            a_df, b_df, id_col='_id',
            flask_port=5006, streamlit_port=8502,
            flask_host='127.0.0.1'
        )

        assert labeler._a_df is a_df
        assert labeler._b_df is b_df
        assert labeler._id_col == '_id'
        assert labeler._flask_port == 5006
        assert labeler._streamlit_port == 8502
        assert labeler._flask_host == '127.0.0.1'
        assert labeler._all_fields is None
        assert labeler._current_fields is None
        assert labeler._server_started is False

    def test_webui_labeler_get_row(self, spark_session, a_df, b_df):
        """Test WebUILabeler _get_row method."""
        labeler = WebUILabeler(a_df, b_df)

        row = labeler._get_row(a_df, 10)
        assert isinstance(row, dict)
        assert row['_id'] == 10

        with pytest.raises(KeyError, match='No row'):
            labeler._get_row(a_df, 999)

    def test_webui_labeler_flask_routes(self, spark_session, a_df, b_df):
        """Test WebUILabeler Flask route setup."""
        labeler = WebUILabeler(a_df, b_df)

        routes = [
            rule.rule for rule in labeler._flask_app.url_map.iter_rules()
        ]
        assert '/get_pair' in routes
        assert '/submit_label' in routes
        assert '/update_fields' in routes
        assert '/done' in routes

    def test_webui_labeler_get_pair_route_waiting(
        self, spark_session, a_df, b_df
    ):
        """Test WebUILabeler get_pair route when no pair is set."""
        labeler = WebUILabeler(a_df, b_df)

        with labeler._flask_app.test_client() as client:
            response = client.get('/get_pair')
            assert response.status_code == 204

    def test_webui_labeler_get_pair_route_with_pair(
        self, spark_session, a_df, b_df
    ):
        """Test WebUILabeler get_pair route when pair is set."""
        labeler = WebUILabeler(a_df, b_df)

        row1 = labeler._get_row(a_df, 10)
        row2 = labeler._get_row(b_df, 20)
        labeler._current_pair = (row1, row2)
        labeler._current_fields_mem = set(row1.keys())

        with labeler._flask_app.test_client() as client:
            response = client.get('/get_pair')
            assert response.status_code == 200
            data = response.get_json()
            assert 'row1' in data
            assert 'row2' in data
            assert 'fields' in data
            assert data['row1']['_id'] == 10
            assert data['row2']['_id'] == 20

    def test_webui_labeler_submit_label_route(self, spark_session, a_df, b_df):
        """Test WebUILabeler submit_label route."""
        labeler = WebUILabeler(a_df, b_df)

        with labeler._flask_app.test_client() as client:
            response = client.post(
                '/submit_label',
                json={'label': 1.0},
                content_type='application/json'
            )
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'ok'

            with labeler._lock:
                assert labeler._label == 1.0

    def test_webui_labeler_update_fields_route(
        self, spark_session, a_df, b_df
    ):
        """Test WebUILabeler update_fields route."""
        labeler = WebUILabeler(a_df, b_df)

        new_fields = ['_id', 'a_attr']

        with labeler._flask_app.test_client() as client:
            response = client.post(
                '/update_fields',
                json={'fields': new_fields},
                content_type='application/json'
            )
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'ok'

            with labeler._lock:
                assert labeler._current_fields == set(new_fields)
                assert labeler._current_fields_mem == set(new_fields)

    def test_webui_labeler_done_route(self, spark_session, a_df, b_df):
        """Test WebUILabeler /done route before and after finish()."""
        labeler = WebUILabeler(a_df, b_df)

        with labeler._flask_app.test_client() as client:
            response = client.get('/done')
            assert response.status_code == 200
            data = response.get_json()
            assert data['done'] is False

            labeler._done = True
            labeler._done_stopped = False
            labeler._n_labeled = 7

            response = client.get('/done')
            data = response.get_json()
            assert data['done'] is True
            assert data['stopped'] is False
            assert data['n_labeled'] == 7

    def test_webui_labeler_streamlit_app_code(self, spark_session, a_df, b_df):
        """Test WebUILabeler _streamlit_app_code generation."""
        labeler = WebUILabeler(a_df, b_df)

        app_code = labeler._streamlit_app_code()

        assert isinstance(app_code, str)
        assert 'streamlit' in app_code
        assert 'FLASK_URL' in app_code
        assert str(labeler._flask_host) in app_code
        assert str(labeler._flask_port) in app_code
        assert 'MatchFlow Web Labeler' in app_code

    @patch.object(WebUILabeler, '_ensure_server_started')
    def test_webui_labeler_call(
        self, mock_ensure_server, spark_session, a_df, b_df
    ):
        """Test WebUILabeler __call__ method."""
        import threading

        labeler = WebUILabeler(a_df, b_df)

        def submit_label_after_delay():
            time.sleep(0.2)
            while True:
                try:
                    with labeler._lock:
                        if labeler._current_pair is not None:
                            labeler._label = 1.0
                            break
                except Exception:
                    pass
                time.sleep(0.05)

        label_thread = threading.Thread(target=submit_label_after_delay)
        label_thread.daemon = True
        label_thread.start()

        result = labeler(10, 20)

        mock_ensure_server.assert_called_once()

        assert result == 1.0

        with labeler._lock:
            assert labeler._current_pair is None
            assert labeler._label is None

        label_thread.join(timeout=2.0)

    @patch.object(WebUILabeler, '_ensure_server_started')
    def test_webui_labeler_call_initializes_fields(
        self, mock_ensure_server, spark_session, a_df, b_df
    ):
        """Test WebUILabeler __call__ initializes fields."""
        import threading

        labeler = WebUILabeler(a_df, b_df)

        assert labeler._all_fields is None
        assert labeler._current_fields is None

        def submit_label():
            time.sleep(0.2)
            while True:
                try:
                    with labeler._lock:
                        if labeler._current_pair is not None:
                            labeler._label = 0.0
                            break
                except Exception:
                    pass
                time.sleep(0.05)

        label_thread = threading.Thread(target=submit_label)
        label_thread.daemon = True
        label_thread.start()

        labeler(10, 20)

        assert labeler._all_fields is not None
        assert isinstance(labeler._all_fields, list)
        assert labeler._current_fields is not None
        assert isinstance(labeler._current_fields, set)

        label_thread.join(timeout=2.0)

    @patch.object(WebUILabeler, '_ensure_server_started')
    def test_webui_labeler_call_different_labels(
        self, mock_ensure_server, spark_session, a_df, b_df
    ):
        """Test WebUILabeler __call__ with different label values."""
        import threading

        labeler = WebUILabeler(a_df, b_df)

        def submit_label_0():
            time.sleep(0.2)
            while True:
                try:
                    with labeler._lock:
                        if labeler._current_pair is not None:
                            labeler._label = 0.0
                            break
                except Exception:
                    pass
                time.sleep(0.05)

        label_thread = threading.Thread(target=submit_label_0)
        label_thread.daemon = True
        label_thread.start()

        result = labeler(10, 20)
        assert result == 0.0

        label_thread.join(timeout=2.0)

        def submit_label_2():
            time.sleep(0.2)
            while True:
                try:
                    with labeler._lock:
                        if labeler._current_pair is not None:
                            labeler._label = 2.0
                            break
                except Exception:
                    pass
                time.sleep(0.05)

        label_thread = threading.Thread(target=submit_label_2)
        label_thread.daemon = True
        label_thread.start()

        result = labeler(11, 21)
        assert result == 2.0

        label_thread.join(timeout=2.0)

