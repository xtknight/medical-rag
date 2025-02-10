import os
import json
import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the directory to walk through
#root_dir = "/data/corpus/pmc_crawl/out"
root_dir = "docs"

COUNT = 100000
FILES = set()
BATCH_SIZE = 256  # Batch size for embedding and insertion

# Walk through the directory recursively
for root, dirs, files in os.walk(root_dir):
    for fn in files:
        # Print the full file path
        file_path = os.path.join(root, fn)
        FILES.add(file_path)

        if len(FILES) >= COUNT:
            break

# DEBUG
#FILES = ['/data/corpus/pmc_crawl/out/PMC5570836.txt']

def get_title(f):
    article_id = f.split('/')[-1].lower().replace('pmc', '').replace('.txt', '').strip()
    #for ln in t.split('\n'):
    #    ln = ln.strip()
    #    if not ln:
    #        continue
    #    return ln.strip()
    try:
        with open('/mnt/data/corpus/pmc_crawl/meta_crawl/meta_mining/out/%s.json' % article_id, 'r', encoding='utf-8') as fd:
            x = json.loads(fd.read())
            uid = str(x['result']['uids'][0])
            return x['result'][uid]['title'].strip()
    except Exception as e:
        print(f, 'error getting title', e)
        return '' # error getting title

def preprocess(t):
    OUT = []
    for ln in t.split('\n'):
        ln = ln.strip()
        if not ln:
            continue
        OUT.append(ln)

    OUT2 = []
    if len(OUT) >= 1:
        OUT2.append(OUT[0])

    for ln_idx in range(1, len(OUT)):
        prev_ln = OUT[ln_idx-1]
        this_ln = OUT[ln_idx]

        #print('PREV_LN', prev_ln)
        #print('THIS_LN', this_ln)

        if len(prev_ln) >= 2 and len(this_ln) >= 1:
            # hyphenated last word
            if prev_ln[-2].isalnum() and prev_ln[-1] == '-' and this_ln[0].isalnum():
                #print('DETECTED')

                #print('OLD PREV', prev_ln)
                #print('OLD THIS', this_ln)

                prev_ln_split = prev_ln.split(' ')
                this_ln_split = this_ln.split(' ')

                # move word to current this line
                last_word = prev_ln_split[-1]
                prev_ln = ' '.join(prev_ln_split[:-1]).strip()
                #OUT[ln_idx] = ' '.join(OUT[ln_idx])

                this_word = this_ln_split[0]
                assert last_word.endswith('-')
                this_word = last_word[:-1].strip() + this_word.strip()

                this_ln = ' '.join([this_word] + this_ln_split[1:]).strip()

                #print('NEW PREV', prev_ln)
                #print('NEW THIS', this_ln)

        OUT2[-1] = prev_ln
        OUT2.append(this_ln)

    return '\n'.join(OUT2).replace('\n', ' ')

FILE_NAMES = []
DOCS = []
TITLES = []

for f in tqdm.tqdm(sorted(FILES)):
    #print(f)
    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
        x = fd.read()
        title = get_title(f) #TODO: get metadata
        preprocessed_text = preprocess(x)
        #print(preprocessed_text)
        FILE_NAMES.append(f)
        TITLES.append(title)
        DOCS.append(preprocessed_text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    separators=[".", "\n"])

'''
separators=[
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    "",
]'''

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

def emb_texts(texts):
    return embedding_model.encode(texts, normalize_embeddings=True).tolist()

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print('Embedding dim', embedding_dim)
#print(test_embedding[:10])



from pymilvus import MilvusClient
milvus_client = MilvusClient(uri="./serotonin.db")


collection_name = "rag_collection"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

_id = 0
for d_idx, doc in enumerate(tqdm.tqdm(DOCS, desc="Creating embeddings")):
    fn = os.path.basename(FILE_NAMES[d_idx])
    title = TITLES[d_idx]
    chunks = text_splitter.create_documents([doc])
    text_lines = [chunk.page_content for chunk in chunks]

    '''
    data = []
    #for i, line in enumerate(tqdm.tqdm(text_lines, desc="Creating embeddings")):
    for i, line in enumerate(text_lines):
        data.append({"id": i, "vector": emb_text(line), "text": line, "source_doc": fn})
    insert_res = milvus_client.insert(collection_name=collection_name, data=data)
    print('Inserted', insert_res["insert_count"], 'rows')
    '''

    # Process in batches
    for i in range(0, len(text_lines), BATCH_SIZE):
        batch_lines = text_lines[i:(i + BATCH_SIZE)]
        batch_embeddings = emb_texts(batch_lines)
        data = []
        for j, (embedding, line) in enumerate(zip(batch_embeddings, batch_lines)):
            data.append({"id": _id, "vector": embedding, "text": line, "source_doc": fn, "title": title})
            _id += 1

        insert_res = milvus_client.insert(collection_name=collection_name, data=data)
        print(f"Inserted {insert_res['insert_count']} rows from file {fn}")
