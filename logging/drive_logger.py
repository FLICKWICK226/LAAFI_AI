"""
LAAFI_AI — Drive Experiment Logger
===================================
Logs model run metrics directly into the Google Doc
"04_Metriques_et_Suivi_Experiments" without manual editing.

Usage
-----
    from drive_logger import ExperimentLogger, ExperimentResult

    logger = ExperimentLogger()

    result = ExperimentResult(
        exp_id      = "EXP-003",
        method      = "MacenkoNormalizer via torchstain",
        dataset     = "PCam val set",
        auc_roc     = 0.9581,
        recall      = 0.812,
        precision   = 0.961,
        accuracy    = 0.889,
        ece         = 0.042,
        failure_mode= "Artéfacts bleus observés sur ~6% des patches éosine",
        note        = "Gain +3.3 pp recall vs EXP-001",
        extra       = {                       # champs spécifiques à l'exp
            "SSIM source": "0.923",
            "FPS à l'inférence": "4.0",
            "Delta recall vs. EXP-001": "+3.3 pp",
        },
    )
    logger.log(result)

Authentication
--------------
The script uses a Google service account or OAuth credentials.
See README_LOGGER.md for setup instructions.

Environment variables (or .env file):
    GDOCS_CREDENTIALS_PATH  path to credentials.json
    GDOCS_TOKEN_PATH        path to token.json (OAuth only)
    LAAFI_DOC_ID            Google Doc ID of the metrics document
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Optional MLflow integration ───────────────────────────────────────────────
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Google API imports ────────────────────────────────────────────────────────
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/documents"]

# Table index in the Google Doc (1-based, matches EXP number)
# Table 1 = Baseline metrics (skip)
# Tables 2-8 = EXP-001 to EXP-007
# Table 9 = Dashboard (auto-updated)
EXP_TABLE_MAP: dict[str, int] = {
    "EXP-001": 2,
    "EXP-002": 3,
    "EXP-003": 4,
    "EXP-004": 5,
    "EXP-005": 6,
    "EXP-006": 7,
    "EXP-007": 8,
}

DASHBOARD_TABLE_INDEX = 9  # "Tableau de bord des cibles"

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """
    Holds one experiment run's results.

    Required fields
    ---------------
    exp_id      : "EXP-001" … "EXP-007"
    method      : short description of the technique tested
    dataset     : dataset name (e.g. "PCam val set")
    auc_roc     : float [0, 1]
    recall      : float [0, 1]  — Sensibilité
    precision   : float [0, 1]

    Optional fields
    ---------------
    accuracy    : float [0, 1]
    ece         : float [0, 1]  — Expected Calibration Error
    failure_mode: observed failure mode (1-2 sentences)
    note        : free-text decision / conclusion
    mlflow_run_id: MLflow run ID if available
    extra       : dict of additional key-value pairs specific to the experiment
                  (e.g. {"SSIM source": "0.923", "Température T apprise": "1.42"})
    """
    exp_id:        str
    method:        str
    dataset:       str
    auc_roc:       float
    recall:        float
    precision:     float

    accuracy:      float | None = None
    ece:           float | None = None
    failure_mode:  str          = "Aucun observé"
    note:          str          = ""
    mlflow_run_id: str          = ""
    extra:         dict[str, str] = field(default_factory=dict)

    def fmt(self, v: float | None, pct: bool = False, decimals: int = 4) -> str:
        if v is None:
            return "—"
        if pct:
            return f"{v * 100:.2f}%"
        return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Google Docs helpers
# ─────────────────────────────────────────────────────────────────────────────

class DriveLogger:
    """Low-level Google Docs API wrapper."""

    def __init__(self, doc_id: str, credentials_path: str, token_path: str = "token.json"):
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google API libraries not installed.\n"
                "Run: pip install google-api-python-client google-auth-httplib2 "
                "google-auth-oauthlib"
            )
        self.doc_id = doc_id
        self.service = self._build_service(credentials_path, token_path)

    def _build_service(self, credentials_path: str, token_path: str):
        creds_file = Path(credentials_path)
        if not creds_file.exists():
            raise FileNotFoundError(
                f"Credentials file not found: {credentials_path}\n"
                "See README_LOGGER.md for setup instructions."
            )

        with open(creds_file) as f:
            creds_data = json.load(f)

        # ── Service account (preferred for automation) ────────────────────
        if creds_data.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=SCOPES
            )
            return build("docs", "v1", credentials=creds)

        # ── OAuth 2.0 (for interactive use) ───────────────────────────────
        creds = None
        token = Path(token_path)
        if token.exists():
            creds = Credentials.from_authorized_user_file(str(token), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(token, "w") as f:
                f.write(creds.to_json())

        return build("docs", "v1", credentials=creds)

    def get_document(self) -> dict:
        return self.service.documents().get(documentId=self.doc_id).execute()

    def _get_table_cells(self, doc: dict, table_index: int) -> list[dict]:
        """Return all rows of the Nth table (1-based index)."""
        tables = []
        for elem in doc["body"]["content"]:
            if "table" in elem:
                tables.append(elem["table"])
        if table_index < 1 or table_index > len(tables):
            raise ValueError(f"Table index {table_index} out of range (doc has {len(tables)} tables)")
        return tables[table_index - 1]["tableRows"]

    def _cell_start_index(self, cell: dict) -> int:
        """Return the start index of the first paragraph in a cell."""
        return cell["content"][0]["startIndex"]

    def _cell_end_index(self, cell: dict) -> int:
        """Return the end index of the last element in a cell."""
        contents = cell["content"]
        last = contents[-1]
        if "paragraph" in last:
            elements = last["paragraph"].get("elements", [])
            if elements:
                return elements[-1]["endIndex"]
        return last.get("endIndex", last["startIndex"] + 1)

    def _cell_text(self, cell: dict) -> str:
        text = ""
        for c in cell.get("content", []):
            for pe in c.get("paragraph", {}).get("elements", []):
                text += pe.get("textRun", {}).get("content", "")
        return text.strip()

    def update_table_cell(
        self,
        table_index: int,
        row_index: int,
        col_index: int,
        new_text: str,
        doc: dict | None = None,
    ) -> None:
        """
        Replace the content of a specific table cell.
        Skips update if the cell already has the target text.
        """
        if doc is None:
            doc = self.get_document()

        rows = self._get_table_cells(doc, table_index)
        cell = rows[row_index]["tableCells"][col_index]
        current_text = self._cell_text(cell)

        # Don't overwrite if already filled (unless it's a placeholder)
        if current_text not in ("À compléter", "", "—") and current_text == new_text:
            return

        start = self._cell_start_index(cell)
        end   = self._cell_end_index(cell)

        # Build batchUpdate requests: delete existing text, insert new text
        requests = []

        # Delete existing content (keep the paragraph marker by using end-1)
        if end > start + 1:
            requests.append({
                "deleteContentRange": {
                    "range": {
                        "segmentId": "",
                        "startIndex": start,
                        "endIndex": end - 1,
                    }
                }
            })
            # After deletion the start index is the same
            insert_index = start
        else:
            insert_index = start

        # Insert new text
        requests.append({
            "insertText": {
                "location": {"segmentId": "", "index": insert_index},
                "text": new_text,
            }
        })

        self.service.documents().batchUpdate(
            documentId=self.doc_id,
            body={"requests": requests},
        ).execute()

    def find_row_by_label(self, rows: list, label: str) -> int | None:
        """Find the row index where column 0 text matches label."""
        for i, row in enumerate(rows):
            cells = row.get("tableCells", [])
            if cells:
                cell_text = self._cell_text(cells[0])
                if cell_text.strip() == label.strip():
                    return i
        return None


# ─────────────────────────────────────────────────────────────────────────────
# High-level Experiment Logger
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Log experiment results into the LAAFI_AI metrics Google Doc.

    Parameters
    ----------
    doc_id           : Google Doc ID (default: from LAAFI_DOC_ID env var)
    credentials_path : Path to credentials.json (default: from GDOCS_CREDENTIALS_PATH)
    token_path       : Path to token.json for OAuth (default: token.json)
    dry_run          : If True, print what would be written without calling the API
    """

    BASELINE = {
        "auc_roc":   0.9513,
        "recall":    0.7492,
        "precision": 0.9629,
        "accuracy":  0.8602,
    }

    def __init__(
        self,
        doc_id: str | None = None,
        credentials_path: str | None = None,
        token_path: str = "token.json",
        dry_run: bool = False,
    ):
        self.doc_id = doc_id or os.environ.get("LAAFI_DOC_ID", "")
        self.credentials_path = credentials_path or os.environ.get(
            "GDOCS_CREDENTIALS_PATH", "credentials.json"
        )
        self.token_path = token_path
        self.dry_run = dry_run

        if not self.doc_id:
            raise ValueError(
                "LAAFI_DOC_ID is not set. "
                "Pass doc_id= or set the LAAFI_DOC_ID environment variable."
            )

        if not dry_run:
            self.driver = DriveLogger(self.doc_id, self.credentials_path, self.token_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, result: ExperimentResult) -> None:
        """
        Write experiment results to the corresponding table in the Google Doc
        and update the dashboard table.
        """
        if result.exp_id not in EXP_TABLE_MAP:
            raise ValueError(
                f"Unknown exp_id '{result.exp_id}'. "
                f"Valid values: {list(EXP_TABLE_MAP.keys())}"
            )

        table_idx = EXP_TABLE_MAP[result.exp_id]
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Build the payload: label → value mapping
        payload = self._build_payload(result, timestamp)

        if self.dry_run:
            self._print_dry_run(result, table_idx, payload)
            return

        print(f"[LAAFI Logger] Writing {result.exp_id} → Doc table {table_idx}...")

        doc = self.driver.get_document()
        rows = self.driver._get_table_cells(doc, table_idx)

        # Write each key-value pair into the correct row
        writes = 0
        for label, value in payload.items():
            row_idx = self.driver.find_row_by_label(rows, label)
            if row_idx is None:
                print(f"  [WARN] Label not found in table: '{label}' — skipping")
                continue
            # Re-fetch doc before each write to get fresh indices
            if writes > 0:
                doc = self.driver.get_document()
                rows = self.driver._get_table_cells(doc, table_idx)
            self.driver.update_table_cell(table_idx, row_idx, 1, value, doc)
            print(f"  [{result.exp_id}] {label} → {value}")
            writes += 1

        # Write extra fields (experiment-specific)
        for label, value in result.extra.items():
            doc = self.driver.get_document()
            rows = self.driver._get_table_cells(doc, table_idx)
            row_idx = self.driver.find_row_by_label(rows, label)
            if row_idx is None:
                print(f"  [WARN] Extra label not found: '{label}' — skipping")
                continue
            self.driver.update_table_cell(table_idx, row_idx, 1, str(value), doc)
            print(f"  [{result.exp_id}] {label} → {value}")

        # Update dashboard
        self._update_dashboard(result)

        # Mirror to MLflow if available
        if MLFLOW_AVAILABLE and mlflow.active_run():
            self._log_to_mlflow(result)

        print(f"[LAAFI Logger] Done. {result.exp_id} logged successfully.")

    def log_from_dict(self, exp_id: str, data: dict[str, Any]) -> None:
        """
        Convenience method to log from a plain dict.
        Keys must match ExperimentResult field names.
        """
        result = ExperimentResult(exp_id=exp_id, **data)
        self.log(result)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_payload(self, r: ExperimentResult, timestamp: str) -> dict[str, str]:
        """Build the label→value mapping for the standard fields."""
        delta_recall = (
            f"{(r.recall - self.BASELINE['recall']) * 100:+.2f} pp"
            if r.recall is not None else "—"
        )
        delta_auc = (
            f"{(r.auc_roc - self.BASELINE['auc_roc']):+.4f}"
            if r.auc_roc is not None else "—"
        )

        base: dict[str, str] = {
            "Date":                 timestamp,
            "Méthode testée":       r.method,
            "Dataset":              r.dataset,
            "AUC ROC":              r.fmt(r.auc_roc),
            "Sensibilité (Recall)": r.fmt(r.recall, pct=True),
            "Précision":            r.fmt(r.precision, pct=True),
        }

        if r.accuracy is not None:
            base["Accuracy"] = r.fmt(r.accuracy, pct=True)
        if r.ece is not None:
            base["ECE avant calibration"] = r.fmt(r.ece, pct=True)
            base["ECE après calibration"] = r.fmt(r.ece, pct=True)
            base["ECE"] = r.fmt(r.ece, pct=True)

        base["Delta recall vs. baseline"] = delta_recall
        base["Delta vs. baseline"]        = f"AUC {delta_auc} | Recall {delta_recall}"
        base["Delta recall vs. EXP-001"]  = delta_recall
        base["Delta vs. EXP-003 (Macenko)"] = delta_recall
        base["Delta vs. ResNet50 actuel"] = f"AUC {delta_auc} | Recall {delta_recall}"

        base["Mode d'échec observé"]     = r.failure_mode
        base["Décision"]                 = r.note if r.note else "À compléter"

        if r.mlflow_run_id:
            base["MLflow run ID"] = r.mlflow_run_id

        return base

    def _update_dashboard(self, r: ExperimentResult) -> None:
        """Update the 'Atteint' column in the dashboard table."""
        if r.auc_roc is None or r.recall is None:
            return

        doc = self.driver.get_document()
        rows = self.driver._get_table_cells(doc, DASHBOARD_TABLE_INDEX)

        updates = {
            "Sensibilité":  f"{r.recall * 100:.2f}% ({r.exp_id})",
            "AUC ROC":      f"{r.auc_roc:.4f} ({r.exp_id})",
        }
        if r.accuracy:
            updates["Accuracy"] = f"{r.accuracy * 100:.2f}% ({r.exp_id})"
        if r.ece:
            updates["ECE (calibration)"] = f"{r.ece * 100:.2f}% ({r.exp_id})"

        for label, value in updates.items():
            row_idx = self.driver.find_row_by_label(rows, label)
            if row_idx is None:
                continue
            doc = self.driver.get_document()
            rows = self.driver._get_table_cells(doc, DASHBOARD_TABLE_INDEX)
            # Write into column index 5 ("Atteint")
            self.driver.update_table_cell(DASHBOARD_TABLE_INDEX, row_idx, 5, value, doc)
            print(f"  [Dashboard] {label} → {value}")

    def _log_to_mlflow(self, r: ExperimentResult) -> None:
        """Mirror key metrics to the active MLflow run."""
        metrics = {
            "auc_roc":   r.auc_roc,
            "recall":    r.recall,
            "precision": r.precision,
        }
        if r.accuracy is not None:
            metrics["accuracy"] = r.accuracy
        if r.ece is not None:
            metrics["ece"] = r.ece
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.set_tag("exp_id", r.exp_id)
        mlflow.set_tag("dataset", r.dataset)
        mlflow.set_tag("failure_mode", r.failure_mode)
        print("  [MLflow] Metrics mirrored to active run.")

    def _print_dry_run(self, r: ExperimentResult, table_idx: int, payload: dict) -> None:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {r.exp_id} → Google Doc table {table_idx}")
        print(f"{'='*60}")
        for label, value in payload.items():
            print(f"  {label:<35} → {value}")
        if r.extra:
            print("  --- extra fields ---")
            for k, v in r.extra.items():
                print(f"  {k:<35} → {v}")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    """
    Quick CLI for logging a result without writing Python.

    python drive_logger.py \
        --exp EXP-003 \
        --method "MacenkoNormalizer" \
        --dataset "PCam val set" \
        --auc 0.958 \
        --recall 0.812 \
        --precision 0.961 \
        --accuracy 0.889 \
        --failure "Artéfacts bleus sur ~6% patches eosin" \
        --note "Committer dans main — gain +3.3 pp recall"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LAAFI_AI — Log experiment metrics to Google Drive"
    )
    parser.add_argument("--exp",       required=True, help="EXP-001 … EXP-007")
    parser.add_argument("--method",    required=True, help="Technique testée")
    parser.add_argument("--dataset",   required=True, help="Dataset utilisé")
    parser.add_argument("--auc",       required=True, type=float, help="AUC ROC [0-1]")
    parser.add_argument("--recall",    required=True, type=float, help="Sensibilité [0-1]")
    parser.add_argument("--precision", required=True, type=float, help="Précision [0-1]")
    parser.add_argument("--accuracy",  type=float,    default=None)
    parser.add_argument("--ece",       type=float,    default=None, help="ECE [0-1]")
    parser.add_argument("--failure",   default="Aucun observé")
    parser.add_argument("--note",      default="")
    parser.add_argument("--mlflow-id", default="")
    parser.add_argument("--dry-run",   action="store_true", help="Ne pas écrire dans Drive")
    parser.add_argument(
        "--extra",
        nargs="*",
        metavar="KEY=VALUE",
        help='Champs supplémentaires ex: "SSIM source=0.923" "FPS=4.0"',
    )

    args = parser.parse_args()

    extra = {}
    if args.extra:
        for item in args.extra:
            if "=" in item:
                k, v = item.split("=", 1)
                extra[k.strip()] = v.strip()

    result = ExperimentResult(
        exp_id       = args.exp,
        method       = args.method,
        dataset      = args.dataset,
        auc_roc      = args.auc,
        recall       = args.recall,
        precision    = args.precision,
        accuracy     = args.accuracy,
        ece          = args.ece,
        failure_mode = args.failure,
        note         = args.note,
        mlflow_run_id= args.mlflow_id,
        extra        = extra,
    )

    logger = ExperimentLogger(dry_run=args.dry_run)
    logger.log(result)


if __name__ == "__main__":
    _cli()
