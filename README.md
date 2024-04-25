# BiasKG

This is the repository for **BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models** [paper link here]()

## Generating the knowledge graph

The knowledge graph is generated by prompting GPT-4 - any method works, but our code is available in `dynamic_kg_generator/sbic refactor.ipynb`


## Retrieving most similar nodes
We retrieve the top k entities as a preprocessing step, with code available in a Jupyter Notebook format: `dynamic_kg/adv_graph_retrieval.ipynb`

This is also packaged as a class in `dynamic_kg/vectorstore.py`

For our preprocessed data files, please refer to `kg_benchmark/data`

## Performing experiments

For GPT models, run `kg_benchmark/running_exp_GPT4_stereo.ipynb`

For open source models, please refer to `kg_benchmark/sbatch_bbq.sh` and `kg_benchmark/sbatch_decodingtrust.sh` for example commands. 

## Preprocessed data

Please contact 14cfl@queensu.ca for the 