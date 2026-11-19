# Parallel Wikipedia scraper on SLURM

A small Go scraper ([wikipedia_scraper.go](wikipedia_scraper.go)) that saves the paragraph text of one English Wikipedia article to `output/<title>.txt`, and a Python helper ([job_submitter.py](job_submitter.py)) that submits one SLURM job per URL so several articles are scraped in parallel on an HPC cluster.

## 1. Build the scraper

```bash
module avail go            # list available Go versions on the cluster
module load go/1.22.5
go version

go build -o wikipedia_scraper .
./wikipedia_scraper https://en.wikipedia.org/wiki/Web_scraping   # quick test
```

Dependencies are pinned in `go.mod` / `go.sum` (colly v2.1.0, which builds with Go 1.22).

## 2. Submit a single job by hand

Create `scraper_job.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=WebScraper
#SBATCH --account=<your_account>
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --output=scraper_output.log

module load go/1.22.5
./wikipedia_scraper https://en.wikipedia.org/wiki/Web_scraping
```

```bash
sbatch scraper_job.slurm
squeue -u "$(whoami)"      # or `show_job` on clusters that provide it
cat scraper_output.log
```

## 3. Submit many jobs in parallel

`job_submitter.py` writes a job script per URL into `jobs/`, submits them with `sbatch`, polls until every job finishes, and exits non-zero if any job failed.

```bash
export SLURM_ACCOUNT=<your_account>   # or pass --account

# one article
python job_submitter.py https://en.wikipedia.org/wiki/Web_scraping

# several articles, up to 3 jobs tracked at once
python job_submitter.py \
  https://en.wikipedia.org/wiki/Wombat \
  https://en.wikipedia.org/wiki/Koala \
  https://en.wikipedia.org/wiki/Kangaroo \
  --max-workers 3
```

Run `python job_submitter.py --help` for all options (time limit, memory, Go module, poll interval).

## Appendix: SLURM "hello world"

A minimal check that job submission works on a new cluster:

```bash
cat > helloworld.c <<'EOF'
#include <stdio.h>
int main(void) { printf("Hello, World!\n"); return 0; }
EOF
gcc -o helloworld helloworld.c

cat > job.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=MyJob
#SBATCH --account=<your_account>
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096
#SBATCH --cpus-per-task=1
./helloworld
EOF

sbatch job.slurm
cat slurm-<jobid>.out
```
