# deepneuron

Experiments in parallel and distributed computing on an HPC cluster (SLURM, multi-GPU nodes).

| Project | Description |
|---|---|
| [parallel-cnn](parallel-cnn/) | CIFAR-10 CNN training on one GPU vs. multiple GPUs with PyTorch DDP and PyTorch Lightning, with timing results |
| [parallel-webscraper](parallel-webscraper/) | Go Wikipedia scraper plus a Python helper that runs many scrape jobs in parallel through SLURM |

## Quick start

```bash
# CNN training
cd parallel-cnn
pip install -r requirements.txt
python parallel-cnn.py 10 5 --batch_size 64

# Web scraper
cd parallel-webscraper
go build -o wikipedia_scraper .
python job_submitter.py https://en.wikipedia.org/wiki/Koala https://en.wikipedia.org/wiki/Wombat
```

See each folder's README for details.

## License

[MIT](LICENSE)
