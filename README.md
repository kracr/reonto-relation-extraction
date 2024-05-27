# ReOnto- Biomedical relation-extraction
ReOnto is a neuro-symbolic approach where we use ontologies and GNNs to extract relations from sentences.

<h3>Dataset:</h3>
The Biorel dataset can be downloaded from this link: https://drive.google.com/drive/folders/1vw2zIxdSoqT2QALDbRVG6loLsgi2doBG


The ADE dataset can be obtained from this link: https://huggingface.co/datasets/ade_corpus_v2

After downloading the data, put it in the designated folder in the "Data" directory.

To process the ADE dataset, run the ADE.py file, followed by ADEconcat.py, and place the processed files in the "Data" folder.
For the Biorel dataset, run the json1 file for the train, test, and dev sets, and place them in the "Data" folder.

<h3>Embedding:</h3>

GloVe embeddings can be downloaded from this link: https://nlp.stanford.edu/data/glove.6B.zip
<h3>Ontology:</h3>
Download the ontology from the following links and place them in the "Ontology" folder:

Ontology of adverse effect: https://bioportal.bioontology.org/ontologies/OAE

National Drug File NDFRT: https://bioportal.bioontology.org/ontologies/NDFRT

DINTO: https://bioportal.bioontology.org/ontologies/DINTO

MEDLINE: https://bioportal.bioontology.org/ontologies/MEDLINEPLUS

NCI: https://bioportal.bioontology.org/ontologies/NCIT

<h3>Structure:</h3>
The "models/" directory contains baseline models (LSTM, CNN, PCNN, ContextAware) in baselines.py and GPGNN and GPGNN_ONTOLOGY models in our_models.py.
The "parsing/" directory contains APIs to convert graphs into tensors, which can be fed into the models.
The "semanticgraph/" directory contains APIs to construct relation graphs from sentences.
The "utils/" directory contains APIs to load word embeddings, evaluate, and operate the graphs.
The "result/" directory is used to store the trained models and output results on the test set.
The model_param.json file contains hyperparameters for the GPGNN model.
Semantic similarity is used to calculate the similarity score between relation labels and paths.

To train the model, run neuro_train.py.
To test the model, run neuro_test.py.

<h3>Acknowledgements:</h3>
The code has been adapted from the GPGNN repository, and the authors are acknowledged for sharing the code.


    <h1>Citation</h1>
    <p>Please cite:</p>
    <pre>
@InProceedings{10.1007/978-3-031-43421-1_14,
    author = {Jain, Monika and Singh, Kuldeep and Mutharaju, Raghava},
    editor = {Koutra, Danai and Plant, Claudia and Gomez Rodriguez, Manuel and Baralis, Elena and Bonchi, Francesco},
    title = {ReOnto: A Neuro-Symbolic Approach for Biomedical Relation Extraction},
    booktitle = {Machine Learning and Knowledge Discovery in Databases: Research Track},
    year = {2023},
    publisher = {Springer Nature Switzerland},
    address = {Cham},
    pages = {230--247},
    isbn = {978-3-031-43421-1}
}
    </pre>
