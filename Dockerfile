FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy benchmark code (run_benchmark.py is self-contained)
COPY run_benchmark.py .

# Pre-download the dataset at build time so it's baked into the image
RUN python3 -c "import datasets; datasets.load_dataset('stanford-crfm/air-bench-2024', 'default', split='test'); datasets.load_dataset('stanford-crfm/air-bench-2024', 'judge_prompts', split='test')"

# Results volume
VOLUME /app/results

ENTRYPOINT ["python3", "run_benchmark.py"]
CMD ["--help"]
