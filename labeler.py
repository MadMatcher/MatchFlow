from abc import abstractmethod, ABC
import time
from tabulate import tabulate
from pyspark.sql.functions import col
import shutil
import textwrap
import pandas as pd


class Labeler(ABC):

    @abstractmethod
    def __call__(self, id1: int, id2: int):
        """
        label the pair (id1, id2)

        returns
        -------
            float : the label for the pair
        """
        pass


class GoldLabeler(Labeler):

    def __init__(self, gold):
        self._gold = gold

    def __call__(self, id1, id2):
        return 1.0 if (id1, id2) in self._gold else 0.0


class DelayedGoldLabeler(Labeler):

    def __init__(self, gold, delay_secs):
        self._gold = gold
        # the number of seconds that the labeler waits until it outputs the label
        # this is used to simulate human labeling
        self._delay_secs = delay_secs

    def __call__(self, id1, id2):
        time.sleep(self._delay_secs)
        return 1.0 if (id1, id2) in self._gold else 0.0


class CLILabeler(Labeler):
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

    @staticmethod
    def _print_dict(d):
        """Tabulate the key/value pairs of a dict."""
        table = list(d.items())
        print(tabulate(table, headers=('field', 'value'), tablefmt="github"))

class CustomLabeler(Labeler):
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
        