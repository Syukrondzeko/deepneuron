"""Submit one SLURM job per Wikipedia URL, run them in parallel, and wait for them to finish.

Examples:
    python job_submitter.py https://en.wikipedia.org/wiki/Web_scraping
    python job_submitter.py https://en.wikipedia.org/wiki/Wombat \\
        https://en.wikipedia.org/wiki/Koala https://en.wikipedia.org/wiki/Kangaroo --max-workers 3
"""

import argparse
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
{account_line}#SBATCH --ntasks=1
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={mem}
#SBATCH --output={log_file}

{module_line}{binary} "{url}"
"""

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def slug(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", url.rstrip("/").rsplit("/", 1)[-1])[:50] or "page"


def job_state(job_id: str) -> str:
    """Return the job's state, preferring squeue (running) and falling back to sacct (finished)."""
    out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True).stdout.strip()
    if out:
        return out
    out = subprocess.run(["sacct", "-n", "-X", "-j", job_id, "-o", "State"],
                         capture_output=True, text=True).stdout.strip()
    # sacct may print e.g. "CANCELLED by 123"; keep the first word only.
    return out.split()[0] if out else "UNKNOWN"


def submit_and_wait(url: str, args) -> tuple[str, str, str, Path]:
    name = slug(url)
    script = args.jobs_dir / f"scraper_{name}.slurm"
    log_file = args.jobs_dir / f"scraper_{name}_%j.log"
    script.write_text(JOB_TEMPLATE.format(
        job_name=f"scrape_{name}",
        account_line=f"#SBATCH --account={args.account}\n" if args.account else "",
        time_limit=args.time,
        mem=args.mem,
        log_file=log_file,
        module_line=f"module load {args.go_module}\n\n" if args.go_module else "",
        binary=args.binary,
        url=url,
    ))

    result = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {url}: {result.stderr.strip()}")
    job_id = result.stdout.strip().split(";")[0]
    print(f"Submitted job {job_id} for {url}")

    state = "PENDING"
    while state not in TERMINAL_STATES:
        time.sleep(args.poll)
        new_state = job_state(job_id)
        if new_state == "UNKNOWN":
            continue  # accounting may lag right after the job leaves the queue
        if new_state != state:
            print(f"Job {job_id} ({name}): {new_state}")
        state = new_state
    return url, job_id, state, Path(str(log_file).replace("%j", job_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="+", help="Wikipedia article URLs to scrape")
    parser.add_argument("--max-workers", type=int, default=4, help="Jobs to submit/track concurrently (default: 4)")
    parser.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT"),
                        help="SLURM account (default: $SLURM_ACCOUNT)")
    parser.add_argument("--binary", default="./wikipedia_scraper", help="Path to the compiled scraper")
    parser.add_argument("--go-module", default="go/1.22.5", help="Environment module to load ('' to skip)")
    parser.add_argument("--time", default="01:00:00", help="Job time limit")
    parser.add_argument("--mem", default="4096", help="Memory per CPU in MB")
    parser.add_argument("--poll", type=int, default=10, help="Seconds between status checks")
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"), help="Where job scripts and logs go")
    args = parser.parse_args()

    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(submit_and_wait, url, args) for url in args.urls]
        for future in as_completed(futures):
            try:
                url, job_id, state, log = future.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going with the other jobs
                failed += 1
                print(f"ERROR: {exc}")
                continue
            if state != "COMPLETED":
                failed += 1
            print(f"Job {job_id} finished with state {state} ({url}); log: {log}")

    print(f"Done: {len(args.urls) - failed}/{len(args.urls)} jobs completed successfully.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
