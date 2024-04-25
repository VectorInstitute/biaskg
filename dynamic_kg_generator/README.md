## DynamicKG generator

For SBIC data, please download the data from this [google drive](https://drive.google.com/drive/folders/12tvKVVfbg12ZUjGMoWjPEZzBjLmYnj7C?usp=sharing)

Files of importance:
- vectorstore.py - a class containing the main methods to load the vector store and perform retrieval
- calculate_test.ipynb - retrieving the BBQ outputs and recalculating the deception rate and no match scores based on the IDs in `train_test_ids_0.1/`
- sbic refactor.ipynb - the notebook containing the prompt for generating the knowledge graph