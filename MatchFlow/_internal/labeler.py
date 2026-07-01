"""
Labeler base class and implementations.

This module is part of the internal implementation and should not be imported directly.
Use the public API in the root package instead.
"""

from abc import abstractmethod, ABC
import time
from tabulate import tabulate
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col
import shutil
import textwrap
import pandas as pd
import os
import atexit
import signal
import socket
import threading
import logging
import tempfile
import subprocess
from flask import Flask, jsonify, request

from .utils import get_logger

log = get_logger(__name__)


class Labeler(ABC):
    """
    Base class for labelers.
    """
    @abstractmethod
    def __call__(self, id1: int, id2: int):
        """
        label the pair (id1, id2)

        returns
        -------
            float : the label for the pair
        """
        pass

    def finish(self, n_labeled: int, stopped: bool = False):
        """Called by label_pairs when labeling is complete."""
        pass


class GoldLabeler(Labeler):
    """
    Gold labeler for labeling pairs of records.

    Parameters
    ----------
    gold : Union[pd.DataFrame, SparkDataFrame]
        the gold dataframe, should contain columns 'id1' and 'id2'
    """
    def __init__(self, gold):
        if isinstance(gold, SparkDataFrame):
            gold = gold.toPandas()
        self._gold = set(zip(gold['id1'], gold['id2']))

    def __call__(self, id1, id2):
        return 1.0 if (id1, id2) in self._gold else 0.0


class DelayedGoldLabeler(Labeler):
    """
    Delayed gold labeler for labeling pairs of records.

    Parameters
    ----------
    gold : Union[pd.DataFrame, SparkDataFrame]
        the gold dataframe, should contain columns 'id1' and 'id2'
    delay_secs : int
        the number of seconds that the labeler waits until it outputs the label
    """
    def __init__(self, gold, delay_secs):
        if isinstance(gold, SparkDataFrame):
            gold = gold.toPandas()
        self._gold = set(zip(gold['id1'], gold['id2']))
        # the number of seconds that the labeler waits until it outputs the label
        # this is used to simulate human labeling
        self._delay_secs = delay_secs

    def __call__(self, id1, id2):
        time.sleep(self._delay_secs)
        return 1.0 if (id1, id2) in self._gold else 0.0


class CLILabeler(Labeler):
    """
    CLI for labeling pairs of records.

    Parameters
    ----------
    a_df : Union[pd.DataFrame, SparkDataFrame]
        the first dataframe
    b_df : Union[pd.DataFrame, SparkDataFrame]
        the second dataframe
    id_col : str, default '_id'
        the column name of the id column
    """
    def __init__(self, a_df, b_df, id_col: str = '_id'):
        self._a_df = a_df
        self._b_df = b_df
        self._id_col = id_col
        self._all_fields = None  # Will be set on first use
        self._current_fields = None

    def __call__(self, id1, id2):
        # Fetch each row as a dict
        row1 = self._get_row(self._a_df, id1)
        row2 = self._get_row(self._b_df, id2)

        # Initialize fields if not already done
        if self._all_fields is None:
            self._all_fields = list(row1.keys())
        if self._current_fields is None:
            self._current_fields = set(self._all_fields)

        print("Do these refer to the same concept?")
        print("=" * 80)  # Separator line
        self._print_row(row1, row2, fields=self._current_fields)
        print("-" * 80)  # Separator line
        label = None
        while label not in ('y', 'n', 'u'):
            label = input('Enter y[es], n[o], u[nsure], h[elp], or s[top]: ').strip().lower()
            if label.startswith('h'):
                self._help_interactive(row1, row2)
                print("=" * 80)
                self._print_row(row1, row2, fields=self._current_fields)
                print("-" * 80)
                label = None 
            elif label.startswith('s'):
                confirm = input('Are you sure? Enter y[es] or n[o]: ').strip().lower()
                if confirm == 'y':
                    return -1.0
                else:
                    label = None

        print('-----------------------------------------------------------------------------------------------')
        return 1.0 if label == 'y' else 0.0 if label == 'n' else 2.0

    def _help_interactive(self, row1, row2):
        """
        Show the interactive 'all fields' help table (add / remove columns)
        with the same rock-solid fixed-width formatting used in _print_row.
        """
        while True:
            # ---------- 1. Build the raw table data ----------
            table = []
            for idx, field in enumerate(self._all_fields, 1):
                in_current = 'x' if field in self._current_fields else ''
                a_val = str(row1.get(field, ''))
                b_val = str(row2.get(field, ''))
                table.append([str(idx), field, in_current, a_val, b_val])

            headers = ['id', 'all fields', 'current', 'A', 'B']

            # ---------- 2. Decide column widths ----------
            term_w = shutil.get_terminal_size((120, 20)).columns

            id_w = max(len(headers[0]), len(str(len(self._all_fields)))) + 1
            field_w = min(30, max(len(headers[1]), max(len(r[1]) for r in table)))
            current_w = max(len(headers[2]), 1) + 1

            n_cols = 5
            padding = 4 + 3 * (n_cols - 1)       # 16 characters of overhead
            available = term_w - (id_w + field_w + current_w + padding)

            colA_w = max(10, available // 2)
            colB_w = max(10, available - colA_w)
            widths = (id_w, field_w, current_w, colA_w, colB_w)

            # ---------- 3. Helper: wrap one logical row into N physical lines ----------
            def wrap_row(cells):
                wrapped = [
                    textwrap.wrap(cells[i], width=widths[i], break_long_words=True) or ['']
                    for i in range(5)
                ]
                height = max(map(len, wrapped))
                for i in range(5):
                    wrapped[i] += [''] * (height - len(wrapped[i]))

                out = []
                for line_idx in range(height):
                    out.append(
                        "│ {:<{idw}} │ {:<{fw}} │ {:<{cw}} │ {:<{aw}} │ {:<{bw}} │".format(
                            wrapped[0][line_idx], wrapped[1][line_idx], wrapped[2][line_idx],
                            wrapped[3][line_idx], wrapped[4][line_idx],
                            idw=id_w, fw=field_w, cw=current_w, aw=colA_w, bw=colB_w
                        )
                    )
                return out

            # ---------- 4. Borders ----------
            def horiz(left, mid, right):
                return (
                    left +
                    "─" * (id_w + 2) + mid +
                    "─" * (field_w + 2) + mid +
                    "─" * (current_w + 2) + mid +
                    "─" * (colA_w + 2) + mid +
                    "─" * (colB_w + 2) + right
                )

            top = horiz("┌", "┬", "┐")
            sep = horiz("├", "┼", "┤")
            bot = horiz("└", "┴", "┘")

            # ---------- 5. Print the table ----------
            print(top)
            for line in wrap_row(headers):
                print(line)
            print(sep)

            for idx, row in enumerate(table):
                for line in wrap_row(row):
                    print(line)
                if idx < len(table) - 1:
                    print(sep)
            print(bot)

            # ---------- 6. Interaction ----------
            cmd = input("Enter a[dd], r[emove], or e[xit]: ").strip().lower()
            if cmd.startswith('a'):
                idxs = input("Comma-separated indices to add: ").split(',')
                for s in idxs:
                    s = s.strip()
                    if s.isdigit() and 1 <= int(s) <= len(self._all_fields):
                        self._current_fields.add(self._all_fields[int(s) - 1])
                    else:
                        print(f"Bad index: {s}")
            elif cmd.startswith('r'):
                idxs = input("Comma-separated indices to remove: ").split(',')
                for s in idxs:
                    s = s.strip()
                    if s.isdigit() and 1 <= int(s) <= len(self._all_fields):
                        self._current_fields.discard(self._all_fields[int(s) - 1])
                    else:
                        print(f"Bad index: {s}")
            elif cmd.startswith('e'):
                break
            else:
                print("Unknown command. Use a, r, or e.")

    def _print_row(self, row1, row2, fields):
        row1_dict = {k: str(v) for k, v in row1.items() if k in fields}
        row2_dict = {k: str(v) for k, v in row2.items() if k in fields}

        term_w = shutil.get_terminal_size((120, 20)).columns
        field_w = max(len("Field"), max(map(len, row1_dict))) + 2
        remaining = term_w - (field_w + 10) 
        colA_w = remaining // 2
        colB_w = remaining - colA_w
        widths = (field_w, colA_w, colB_w)

        def wrap_row(cells):
            wrapped = [
                textwrap.wrap(cells[i], width=widths[i], break_long_words=True) or ['']
                for i in range(3)
            ]
            height = max(map(len, wrapped))
            # pad short columns out to the full height
            for i in range(3):
                wrapped[i] += [''] * (height - len(wrapped[i]))

            lines = []
            for j in range(height):
                line = "│ {:<{fw}} │ {:<{aw}} │ {:<{bw}} │".format(
                    wrapped[0][j], wrapped[1][j], wrapped[2][j],
                    fw=field_w, aw=colA_w, bw=colB_w
                )
                lines.append(line)
            return lines

        hline = "├" + "─" * (field_w + 2) + "┼" + "─" * (colA_w + 2) + "┼" + "─" * (colB_w + 2) + "┤"
        top = "┌" + "─" * (field_w + 2) + "┬" + "─" * (colA_w + 2) + "┬" + "─" * (colB_w + 2) + "┐"
        bot = "└" + "─" * (field_w + 2) + "┴" + "─" * (colA_w + 2) + "┴" + "─" * (colB_w + 2) + "┘"

        print(top)
        # header
        for line in wrap_row(("Field", "From A", "From B")):
            print(line)
        print(hline)

        # each data row
        for index, key in enumerate(row1_dict):
            a_val = row1_dict[key]
            b_val = row2_dict.get(key, '')
            for line in wrap_row((key, a_val, b_val)):
                print(line)
            # Print hline only if it's not the last row
            if index < len(row1_dict) - 1:
                print(hline)
        print(bot)

    def _get_row(self, df, row_id):
        """Fetch a single row from a DataFrame as a dict."""
        if isinstance(df, pd.DataFrame):
            rows = df[df[self._id_col] == row_id]
            if len(rows) == 0:
                raise KeyError(f"No row with {self._id_col}={row_id}")
            return rows.iloc[0].to_dict()
        else:  # Spark DataFrame
            rows = df.filter(col(self._id_col) == row_id).limit(1).collect()
            if not rows:
                raise KeyError(f"No row with {self._id_col}={row_id}")
            return rows[0].asDict()

    def finish(self, n_labeled: int, stopped: bool = False):
        print()
        if stopped:
            print(f"Labeling stopped. {n_labeled} pair(s) labeled.")
        else:
            print(f"Labeling complete. {n_labeled} pair(s) labeled.")

    @staticmethod
    def _print_dict(d):
        """Tabulate the key/value pairs of a dict."""
        table = list(d.items())
        print(tabulate(table, headers=('field', 'value'), tablefmt="github"))

class CustomLabeler(Labeler):
    """
    Custom labeler for labeling pairs of records.

    Parameters
    ----------
    a_df : Union[pd.DataFrame, SparkDataFrame]
        the first dataframe
    b_df : Union[pd.DataFrame, SparkDataFrame]
        the second dataframe
    id_col : str, default '_id'
        the column name of the id column
    """
    def __init__(self, a_df, b_df, id_col: str = '_id'):
        self._a_df = a_df
        self._b_df = b_df
        self._id_col = id_col


    def __call__(self, id1, id2):
        # fetch each row as a dict
        row1 = self._get_row(self._a_df, id1)
        row2 = self._get_row(self._b_df, id2)
        label = self.label_pair(row1, row2)
        return label

    @abstractmethod
    def label_pair(self, row1, row2):
        """
        label the pair (id1, id2)

        returns
        -------
            float : the label for the pair
        """
        pass

    def _get_row(self, df, row_id):
        """Fetch a single row from a DataFrame as a dict."""
        if isinstance(df, pd.DataFrame):
            rows = df[df[self._id_col] == row_id]
            if len(rows) == 0:
                raise KeyError(f"No row with {self._id_col}={row_id}")
            return rows.iloc[0].to_dict()
        else:  # Spark DataFrame
            rows = df.filter(col(self._id_col) == row_id).limit(1).collect()
            if not rows:
                raise KeyError(f"No row with {self._id_col}={row_id}")
            return rows[0].asDict()

    @staticmethod
    def _print_dict(d):
        """Tabulate the key/value pairs of a dict."""
        table = list(d.items())
        print(tabulate(table, headers=('field', 'value'), tablefmt="github"))
        
class WebUILabeler(Labeler):
    """
    Web interface for labeling pairs of records.

    Parameters
    ----------
    a_df, b_df : Union[pd.DataFrame, SparkDataFrame]
    id_col : str, default '_id'
    flask_port : int, default 5005
    streamlit_port : int, default 8501
    flask_host : str, default '127.0.0.1'   
    """
    def __init__(self, a_df, b_df, id_col: str = '_id', flask_port: int = 5005, streamlit_port: int = 8501, flask_host: str = '127.0.0.1'):
        self._a_df = a_df
        self._b_df = b_df
        self._id_col = id_col
        self._all_fields = None  # Will be set on first use
        self._current_fields = None
        self._flask_port = flask_port
        self._streamlit_port = streamlit_port
        self._flask_host = flask_host
        self._lock = threading.Lock()
        self._current_pair = None
        self._current_fields_mem = None
        self._label = None
        self._flask_app = Flask(__name__)
        # Store the original DataFrame column order
        self._column_order = list(a_df.columns)
        self._setup_flask_routes()
        self._flask_thread = None
        self._streamlit_proc = None
        self._app_path = None
        self._cleaned_up = False
        self._server_started = False
        self._done = False
        self._done_stopped = False
        self._n_labeled = 0
        # Do NOT start servers here

    def _setup_flask_routes(self):
        @self._flask_app.route('/get_pair', methods=['GET'])
        def get_pair():
            with self._lock:
                if self._current_pair is not None:
                    id1 = self._current_pair[0].get(self._id_col, None)
                    id2 = self._current_pair[1].get(self._id_col, None)
                    return jsonify({
                        'row1': self._current_pair[0],
                        'row2': self._current_pair[1],
                        'fields': list(self._current_fields_mem)
                    })
                else:
                    return jsonify({'status': 'waiting'}), 204

        @self._flask_app.route('/submit_label', methods=['POST'])
        def submit_label():
            data = request.get_json()
            with self._lock:
                self._label = data.get('label')
            return jsonify({'status': 'ok'})

        @self._flask_app.route('/update_fields', methods=['POST'])
        def update_fields():
            data = request.get_json()
            with self._lock:
                self._current_fields = set(data.get('fields', []))
                # Also update the memory version for immediate use
                self._current_fields_mem = self._current_fields
            return jsonify({'status': 'ok'})

        @self._flask_app.route('/done', methods=['GET'])
        def done():
            with self._lock:
                return jsonify({
                    'done': self._done,
                    'stopped': self._done_stopped,
                    'n_labeled': self._n_labeled,
                })

    @staticmethod
    def _port_in_use(host, port):
        """Return True if a TCP server is already bound to (host, port)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return False
            except OSError:
                return True

    def _ensure_server_started(self):
        if not self._server_started:
            # Fail fast with an actionable message if a previous run left a
            # server holding one of the ports (e.g. after a crash).
            for name, host, port in (
                ('Flask', self._flask_host, self._flask_port),
                ('Streamlit', '127.0.0.1', self._streamlit_port),
            ):
                if self._port_in_use(host, port):
                    raise RuntimeError(
                        f"{name} port {port} is already in use. A previous "
                        f"labeling session may still be running. Stop it "
                        f"(e.g. `lsof -ti tcp:{port} | xargs kill`), or pass a "
                        f"different port to WebUILabeler."
                    )

            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            self._flask_thread = threading.Thread(
                target=self._flask_app.run,
                kwargs={'host': self._flask_host, 'port': self._flask_port, 'debug': False, 'use_reloader': False},
                daemon=True
            )
            self._flask_thread.start()
            # Launch Streamlit UI as a subprocess
            app_code = self._streamlit_app_code()
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
                f.write(app_code)
                self._app_path = f.name
            streamlit_cmd = ["streamlit", "run", self._app_path, "--server.port", str(self._streamlit_port), "--server.headless", "true"]
            # start_new_session=True puts Streamlit (and any children it spawns)
            # in its own process group so we can reliably kill the whole tree.
            self._streamlit_proc = subprocess.Popen(
                streamlit_cmd, env=os.environ.copy(), start_new_session=True
            )
            # Ensure the subprocess is torn down even if the caller crashes or
            # exits without calling finish(), so the port is not left occupied.
            atexit.register(self._cleanup)
            self._server_started = True
            time.sleep(2)

    def __call__(self, id1, id2):
        self._ensure_server_started()
        row1 = self._get_row(self._a_df, id1)
        row2 = self._get_row(self._b_df, id2)
        if self._all_fields is None:
            self._all_fields = list(row1.keys())
        if self._current_fields is None:
            self._current_fields = set(self._all_fields)
        with self._lock:
            # Block if a previous pair is still waiting for a label
            while self._current_pair is not None:
                self._lock.release()
                time.sleep(0.1)
                self._lock.acquire()
            self._current_pair = (row1, row2)
            # Use the persisted field selection, fallback to all fields if not set
            self._current_fields_mem = self._current_fields if self._current_fields else set(self._all_fields)
            self._label = None
        # Wait for label to be set by UI
        while True:
            with self._lock:
                label = self._label
            if label is not None:
                break
            time.sleep(0.2)
        with self._lock:
            self._current_pair = None
            # Don't clear _current_fields_mem - preserve the field selection
            self._label = None
        return label

    def finish(self, n_labeled: int, stopped: bool = False):
        with self._lock:
            self._done = True
            self._done_stopped = stopped
            self._n_labeled = n_labeled

        # Give the UI a moment to render the final "complete"/"stopped" message
        # before tearing the Streamlit subprocess down.
        if self._streamlit_proc is not None:
            time.sleep(2)
        self._cleanup()

    def _cleanup(self):
        """Terminate the Streamlit subprocess and remove the temp app file.

        Idempotent and safe to call from finish() or an atexit handler, so the
        UI process is not left running (holding its port) if the caller crashes.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True

        proc = self._streamlit_proc
        if proc is not None and proc.poll() is None:
            try:
                # Kill the whole process group created via start_new_session so
                # any child processes Streamlit spawned are cleaned up too.
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait()
        self._streamlit_proc = None

        if self._app_path is not None:
            try:
                os.unlink(self._app_path)
            except OSError:
                pass
            self._app_path = None

    def _get_row(self, df, row_id):
        """Fetch a single row from a DataFrame as a dict."""
        if isinstance(df, pd.DataFrame):
            rows = df[df[self._id_col] == row_id]
            if len(rows) == 0:
                raise KeyError(f"No row with {self._id_col}={row_id}")
            return rows.iloc[0].to_dict()
        else:  # Spark DataFrame
            rows = df.filter(
                col(self._id_col) == row_id
            ).limit(1).collect()
            if not rows:
                raise KeyError(f"No row with {self._id_col}={row_id}")
            return rows[0].asDict()

    def _streamlit_app_code(self):
        column_order = self._column_order
        return f'''
import streamlit as st
import requests
import time
import os
import json
import tempfile
import textwrap
import pandas as pd

st.title("MatchFlow Web Labeler")

FLASK_URL = "http://{self._flask_host}:{self._flask_port}"

if 'last_pair' not in st.session_state:
    st.session_state['last_pair'] = None

if 'selected_fields' not in st.session_state:
    st.session_state['selected_fields'] = {column_order}

try:
    done_resp = requests.get(FLASK_URL + '/done', timeout=5)
    if done_resp.status_code == 200:
        done_data = done_resp.json()
        if done_data.get('done', False):
            n = done_data.get('n_labeled', 0)
            if done_data.get('stopped', False):
                st.info(f"Labeling stopped. {{n}} pair(s) labeled.")
            else:
                st.success(f"Labeling complete! {{n}} pair(s) labeled.")
            st.stop()
except Exception:
    pass

if st.session_state.get('stopped', False):
    st.write("Labeling stopped.")
else:
    pair = None
    try:
        resp = requests.get(FLASK_URL + '/get_pair', timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            row1 = data['row1']
            row2 = data['row2']
            fields = data['fields']
            st.session_state['last_pair'] = (row1, row2, fields)
            pair = (row1, row2, fields)
        if resp.status_code == 204:
            pair = None
    except Exception as e:
        pair = st.session_state['last_pair']

    if pair is not None:
        row1, row2, fields = pair
        st.subheader("Do these two records refer to the same entity?")
        # Interactive field selection
        all_fields = {column_order}
        
        # Function to send field updates to backend
        def update_backend_fields(new_fields):
            try:
                requests.post(FLASK_URL + '/update_fields', json={{'fields': new_fields}}, timeout=10)
            except Exception as e:
                st.error(f"Failed to update fields: {{e}}")
        
        # Use the fields from the backend to ensure consistency
        current_selection = fields
        
        # Create multiselect with callback
        selected_fields = st.multiselect(
            "Fields to display:",
            options=all_fields,
            default=current_selection,
            key='field_selector',
        )
        
        # Send updates to backend when selection changes
        # Convert to sets for proper comparison
        if set(selected_fields) != set(current_selection):
            update_backend_fields(selected_fields)
        def wrap(val, width=40):
            return textwrap.fill(str(val), width=width, break_long_words=False)
        table_data = []
        # Maintain original column order by filtering all_fields to only include selected_fields
        ordered_fields = [f for f in all_fields if f in selected_fields]
        for f in ordered_fields:
            a_val = wrap(row1.get(f, ''), width=40)
            b_val = wrap(row2.get(f, ''), width=40)
            table_data.append((f, a_val, b_val))  # Do NOT wrap f
        # Render as HTML table for perfect wrapping
        table_html = """<table>
<thead>
<tr>
  <th style='white-space:nowrap'>Field</th>
  <th style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>From A</th>
  <th style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>From B</th>
</tr>
</thead>
<tbody>
"""
        for f, a_val, b_val in table_data:
            table_html += f"<tr><td style='white-space:nowrap'>{{f}}</td><td style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>{{a_val}}</td><td style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>{{b_val}}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Add light blue button styling
        st.markdown("""
        <style>
        .stButton > button {{
            background-color: #87CEEB !important;
            color: #000000 !important;
            border: 2px solid #4682B4 !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
        }}
        .stButton > button:hover {{
            background-color: #4682B4 !important;
            color: white !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        if 'label_sent' not in st.session_state:
            st.session_state['label_sent'] = False
        def send_label(label):
            try:
                requests.post(FLASK_URL + '/submit_label', json={{'label': label}}, timeout=10)
                st.session_state['label_sent'] = True
                if label == -1.0:
                    st.session_state['stopped'] = True
            except Exception as e:
                st.error(f"Failed to send label: {{e}}")
        with col1:
            if st.button("Yes (y)", key="yes_btn"):
                send_label(1.0)
        with col2:
            if st.button("No (n)", key="no_btn"):
                send_label(0.0)
        with col3:
            if st.button("Unsure (u)", key="unsure_btn"):
                send_label(2.0)
        with col4:
            if st.button("Stop (s)", key="stop_btn"):
                send_label(-1.0)
        if st.session_state['label_sent']:
            if st.session_state.get('stopped', False):
                st.write("Labeling stopped.")
            else:
                st.success("Label sent! Waiting for next pair...")
                st.session_state['label_sent'] = False
                time.sleep(0.3)  # Give backend time to update
                st.rerun()
    else:
        st.write("No pair to label. Waiting...")
        time.sleep(0.2)
        st.rerun()
'''


