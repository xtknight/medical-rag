# Download PMC medical data
 (PMCxxxx.txt)

# Get documents related to serotonin
grep -ilr serotonin /data/corpus/pmc_crawl/out/ > serotonin.matches.txt

# Copy documents to docs folder
cat serotonin.matches.txt |xargs -I{} cp -v {} ./docs/

# Index using Milvus
python3 index.py

# Query using Milvus and Qwen 2.5-14B
