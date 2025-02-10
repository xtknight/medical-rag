# Processing PMC Medical Data with Milvus and Qwen 2.5-14B

This guide walks through downloading PMC medical data, extracting specific documents, and indexing/querying with Milvus.

## Prerequisites
Llama.cpp (CPU supported)

## Steps for Sample

### 1. Download PMC Medical Data (PubMed publically avaliable data)
 docs/PMCxxxx.txt

### 2. Get documents related to serotonin
```
  grep -ilr serotonin /data/corpus/pmc_crawl/out/ > serotonin.matches.txt
```

### 3. Copy documents to docs folder
```
cat serotonin.matches.txt |xargs -I{} cp -v {} ./docs/
```

### 4. Index using Milvus
```
python3 index.py
```

### 5. Query using Milvus and Qwen 2.5-14B
```
python3 query.py
```
