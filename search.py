import re
import sys
import json
import random
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
milvus_client = MilvusClient(uri="./serotonin.db")

from collections import defaultdict
from llama_cpp import Llama

# min statement length in words
MIN_STATEMENT_LENGTH = 5

print('Starting up...')
llm = Llama(
      model_path="../qwen2.5-14b-instruct-q5_k_m.gguf",
      n_gpu_layers=33, # Uncomment to use GPU acceleration
      seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
      verbose=False,
      n_ctx=4096 # 32K for EXAONE
)

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

print('Initialized.')

question = ''

#SIM_CACHE = []

def calculate_vector_similarity(sentence1, sentence2):
    """
    Calculates the cosine similarity between the embeddings of two sentences.
    """
    embeddings = embedding_model.encode([sentence1, sentence2], normalize_embeddings=True)
    similarity = embeddings[0] @ embeddings[1]  # Dot product for cosine similarity due to normalization
    print('sentence1', sentence1)
    print('sentence2', sentence2)
    print('similarity', similarity)
    return similarity

def get_sources(sentence):
    # be more flexible: sometimes parentheses are used
    #sentence2 = sentence.replace('(', '[')
    #sentence2 = sentence2.replace(')', ']')
    if '[' in sentence and ']' in sentence:
        src_list = sentence.rsplit('[', 1)[1].rsplit(']', 1)[0].strip()
        if src_list.lower() == 'general':
            source = 'general'
        else:
            src_list2 = set()
            for x in src_list.split(','):
                if x.strip().upper().startswith('PMC'): # valid
                    src_list2.add(x.strip())
            if len(src_list2) == 0:
                return None
            source = ','.join(sorted(src_list2))
        return source
    else:
        return None

def split_sentences(text):
    # Split text based on full stops followed by whitespace or end of string
    sentences = re.split(r'\.\s*', text)
    #sentences = re.split(r'(\.+|\n+)', text)
    # Remove any empty strings from the result
    return [sentence.strip() for sentence in sentences if sentence.strip()]

while question != 'done':
    #question = 'what are serotonin 1A and 2A receptors?'
    #question = 'what is dopamine?'
    question = input('Question: ')

    collection_name = "rag_collection"
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question)],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=5,  # Return top 5 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text", "source_doc", "title"],  # Return the text field
    )
    #print(search_res)

    #print('Results:')
    #for r in search_res:
    #    print(r)

    retrieved_lines_with_distances = [(res["entity"]["title"], res["entity"]["text"], res["distance"], res["entity"]["source_doc"]) for res in search_res[0]]
    print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))
    print()

    #context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
    context = []

    for r in retrieved_lines_with_distances:
        title, text, dist, src = r
        # remove brackets and text within brackets
        text = re.sub(r"\[[^\]]*\]", "", text)
        # remove the fullstop in et al that interferes with preprocessing
        text = text.replace('et al.', 'et al')
        text = text.strip()

        if text.startswith('. '):
            text = text[2:].strip()

        src = src.replace('.txt', '') # .txt gets truncated sometimes during generation

        if len(text) == 0:
            continue

        if not text.endswith('.'):
            text += '. '

        text = text.replace('. ', ' [%s]. ' % src).strip()

        context.append('''    <document>
        <source>%s</source>
        <title>%s</title>
        <text>%s</text>
    </document>''' % (src, title, text))

    context = '\n\n'.join(context)
    real_context = context

    FINAL_STATEMENTS = []
    VALID_STATEMENTS = []

    #EXTRA_TRIES = 5
    #MIN_VERIFICATIONS = 2

    #TOP_KS = [1] + (EXTRA_TRIES*[5])

    GENERAL_KNOWLEDGE = []

    for i in range(2):
        print('Generate', i)

        if i == 0:
            PROMPT = """
            Answer the following question.
            <question>
            {question}
            </question>
            """

            prompt = PROMPT.format(question=question)
        else:
            PROMPT = """
            Use the following documents enclosed in <context> tags to provide an answer to the question enclosed in <question> tags. Each document is enclosed in a <document> tag with <title>, <source>, <text> tags denoting the title, source, and the contents of the documents. You MUST cite your sources in brackets at the end of EVERY sentence that you generate. If the text is general knowledge, cite the source as "general". If the text is part of a specified document, cite the document txt file.

            For example,
            This is information from SRC123 that was generated [SRC123]. This is common knowledge [general]. This is more common knowledge [general].

            <context>
        {real_context}
            </context>
            <question>
            {question}
            </question>
            """

            prompt = PROMPT.format(real_context=real_context, question=question)

        print('Prompt\n')
        print(prompt)
        print()

        print('Response\n')

        output = llm.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            top_k = 1,
            temperature = 0.9,
            seed = random.randint(0, 100),
            max_tokens = 1024,
            stream = False
        )

        final_output = output['choices'][0]['message']['content']

        print('Output', i, final_output)

        valid_statements = []
        #for sent in final_output.split('. '):
        for sent in split_sentences(final_output):
            #sent = sent.strip()
            #if not sent:
            #    valid_statements.append('') # newline
            #    continue
            sent = sent.replace('(general)', '[general]')
            # (PMC123) -> [PMC123]
            sent = re.sub(r"\(PMC([0-9].*?)\)", r"[PMC\1]", sent)

            if i == 0:
                #valid_statements.append(sent.rstrip('.') + ' [general].')
                if len(sent.strip()) > 0:
                    GENERAL_KNOWLEDGE.append(sent)
            else:
                if len(sent.strip()) > 0:
                    source = get_sources(sent)
                    if source is not None:
                        #valid_statements.append((sent, src))
                        #if len(sent.split()) >= MIN_STATEMENT_LENGTH:
                        valid_statements.append(sent)
                    else:
                        valid_statements.append(sent + ' [general]')
                else:
                    valid_statements.append(sent)

        if i != 0:
            VALID_STATEMENTS.append(valid_statements)
            print('\n\nValidated statements:\n')
            print('\n'.join(valid_statements))

        #print()
        #print()

    print('Merge')
    #for doc_statements_idx, doc_statements in enumerate([VALID_STATEMENTS[0]]): # only use the first top_k=1 quality doc at first

    # guide output based on topk=1
    for doc_statements_idx, doc_statements in enumerate([VALID_STATEMENTS[0]]): # only use the first top_k=1 quality doc at first
        for doc in doc_statements:
            if len(doc.strip()) == 0:
                FINAL_STATEMENTS.append(doc)
                continue
            doc_sources = get_sources(doc)

            #is_general = (doc_sources == 'general')
            is_general = False
            for knowledge_doc in GENERAL_KNOWLEDGE:
                if (calculate_vector_similarity(doc.rsplit('[', 1)[0], knowledge_doc.rsplit('[', 1)[0]) >= 0.85):
                    is_general = True
                    break

            if is_general and doc_sources != 'general':
                # conflict: don't output? or output general knowledge sentence instead?
                print('is_in_general_knowledge && doc_source != general: conflict (converting to general citation):', doc)
                #if ' [PMC' in doc:
                doc = doc.rsplit(' [PMC', 1)[0] + ' [general]'
                #elif ' (PMC' in doc:
                #    doc = doc.rsplit(' (PMC', 1)[0] + ' [general]'
                #else:
                #    assert None, 'unknown citaiton: ' + str(doc)
                #continue

            if doc not in FINAL_STATEMENTS:
                FINAL_STATEMENTS.append(doc)

    print(FINAL_STATEMENTS)
    print('\n\nValidated statements:\n')
    print('. '.join(FINAL_STATEMENTS))

#What are anti-anxiety drugs?
#what happens during dopamine deficiency?
#what happens in a lack of serotonin?
#세로토닌이 부족하면 어떻게 되나요?
