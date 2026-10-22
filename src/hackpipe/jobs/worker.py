"""`python -m hackpipe.jobs.worker <job_id>` — runs one job (started by runner.submit)."""
import sys

from hackpipe.jobs.runner import run_job

if __name__ == "__main__":
    sys.exit(run_job(sys.argv[1]))
