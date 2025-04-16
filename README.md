# rag4rag: Using RAG to generate data for model fine-tuning

**RAG datasets have a synthetic data problem. This repo is an ongoing experiment to solve it.**

Using synthetic data to fine-tune retrieval and embedding models is a cheap and popular way to increase accuracy for RAG and agentic workflows. But synthetic data has a huge hallucination problem. This problem, however, can be fixed by using RAG to generate sythnetic data for fine-tuning rather than using LLMs in isolation. We call this **rag4rag**.

### What type of data is needed for fine-tuning a retrieval model?

Retrieval is the process of accessing or recovering stored information or items. To train models to be better at retrieval, we feed them a `context` along with `question` and `answer` pairs.

Here's an example:

# rag4rag: Using RAG to generate data for fine-tuning retrieval models

**RAG datasets have a synthetic data problem. This repo is an ongoing experiment to solve it.**

Using synthetic data to fine-tune retrieval models is a cheap and popular way to increase accuracy for RAG and agentic workflows. But synthetic data has a huge hallucination problem. This problem, however, can be fixed by using RAG to generate sythnetic data for fine-tuning rather than using LLMs in isolation. We call this **rag4rag**.

### What type of data is needed for fine-tuning a retrieval model?
Retrieval is the process of accessing or recovering stored information or items. To train models to be better at retrieval, we feed them a `context` along with `question` and `answer` pairs.

Here's an example:
```json
{
    "context": "Beyoncé's debut album, Dangerously in Love (2003), established her as a solo artist worldwide.",
    "question": "Which album established Beyoncé as a worldwide artist?",
    "answer": "Dangerously in Love."
}
```

Curating these types of datasets with human annotators is both costly and laborious because it requires humans to read a chunk of text (which can be long), come with their own sets of questions, and then give the answers to those questions. As a result, many have turned to using LLMs to generate this type of data synthetically.

### The hallucination problem

To generate synthetic data, usually the `context` is passed to the LLM in a prompt, along with instructions to use it to generate a `question` and an `answer`. The problem is that when generating the `answer`, the LLM can hallucinate, producing an incorrect `answer` based on information in its parametric memory (aka the training data) rather than an `answer` grounded in the `context`.

Here's a real example I obtained during my research:

Curating these types of datasets with human annotators is both costly and laborious because it requires humans to read a chunk of text (which can be long), come with their own sets of questions, and then give the answers to those questions. As a result, many have turned to using LLMs to generate this type of data synthetically. 

### The hallucination problem

To generate synthetic data, usually the `context` is passed to the LLM in a prompt, along with instructions to use it to generate a `question` and an `answer`. The problem is that when generating the `answer`, the LLM can hallucinate, producing an incorrect `answer` based on information in its parametric memory (aka the training data) rather than an `answer` grounded in the `context`. 

Here's a real example I obtained during my research:
```json
{
    "context": "Beyoncé's debut album, Dangerously in Love (2003), established her as a solo artist worldwide.",
    "question": "Which album established Beyoncé as a worldwide artist?",
    "synthetic_answer": "1989."
}
```

Although the answer is so obviously Crazy In Love, the gave the wrong answer. But the wrong answer it gave is VERY interesting... because 1989 was the album that established Taylor Swift as a worldwide artist. This proves that not only did the model not ground its answer in the provided context, but it relied on its training to produce an incorrect answer. In short, it hallucinated.

### RAG as the solution

In this notebook, we examine the hallucination problem in synthetic RAG data and propose and test a solution: using RAG to generate data to fine-tune retrievers. We find that using RAG, rather than simply instructing a LLM to go back into the context to generate an answer, dramatically improves accuracy.

### Guide to the repo

This repo contains several Jupyter notebooks used to run the experiment and illustrate the results.

1. [01_create_synthetic_rag_dataset_distilabel.ipynb](https://github.com/m-newhauser/rag4rag/blob/main/notebooks/01_create_synthetic_rag_dataset_distilabel.ipynb):
   Create a synthetic RAG dataset with [Distilabel](https://github.com/argilla-io/distilabel).
Although the answer is so obviously Crazy In Love, the gave the wrong answer. But the wrong answer it gave is VERY interesting... because 1989 was the album that established Taylor Swift as a worldwide artist. This proves that not only did the model not ground its answer in the provided context, but it relied on its training to produce an incorrect answer. In short, it hallucinated.

### RAG as the solution
In this notebook, we examine the hallucination problem in synthetic RAG data and propose and test a solution: using RAG to generate data to fine-tune retrievers. We find that using RAG, rather than simply instructing a LLM to go back into the context to generate an answer, dramatically improves accuracy.

### Guide to the repo
This repo contains several Jupyter notebooks used to run the experiment and illustrate the results.
1. [01_create_synthetic_rag_dataset_distilabel.ipynb](https://github.com/m-newhauser/rag4rag/blob/main/notebooks/01_create_synthetic_rag_dataset_distilabel.ipynb): 
Create a synthetic RAG dataset with [Distilabel](https://github.com/argilla-io/distilabel).
2. [02_lettucedetect_on_sythnetic_rag_dataset.ipynb](https://github.com/m-newhauser/rag4rag/blob/main/notebooks/02_lettucedetect_on_sythnetic_rag_dataset.ipynb): Uses LettuceDetect to detect hallucinations in a synthetic RAG dataset created in notebook #1.
3. [03_generate_synthetic_data_with_rag.ipynb](https://github.com/m-newhauser/rag4rag/blob/main/notebooks/03_generate_synthetic_data_with_rag.ipynb): Uses a RAG pipeline (LangChain + FAISS) to generate answer data for retrieval fine-tuning.
4. [rag4rag_experiment_demo.ipynb](https://github.com/m-newhauser/rag4rag/blob/main/notebooks/rag4rag_experiment_demo.ipynb): Full demo notebook that investigates the hallucination problem and compares results of generating synthetic data with a zero-shot LLM approach and the rag4rag solution.
