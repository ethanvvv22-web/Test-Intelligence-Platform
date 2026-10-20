import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("TIP_DATA_DIR", ROOT / "data"))
RAW_BUILDS_DIR = DATA_DIR / "raw" / "builds"
MODELS_DIR = Path(os.environ.get("TIP_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("TIP_REPORTS_DIR", ROOT / "reports"))

ZUUL_API = os.environ.get("TIP_ZUUL_API", "https://zuul.opendev.org/api")
ZUUL_TENANT = os.environ.get("TIP_ZUUL_TENANT", "openstack")

# Results that mean "the job ran and the change was rejected". NODE_FAILURE and
# similar are pure infrastructure outcomes and are not part of the triage task.
FAILURE_RESULTS = ("FAILURE", "TIMED_OUT", "POST_FAILURE", "RETRY_LIMIT")
PIPELINES = ("check", "gate")
