# Chinese Elite Nationalism: Supplemental Code

The code and data in this repository is an example of a reproducible research workflow for the research project "Reading Chinese Elite Nationalism from Foreign Policy Scholarship on China-Japan Relations." 

## Set up Environment
The code is written in Python 3.7.13 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

## Prepare Text Data
Next, you can download the `raw_data` folder and unzip all files wihin the folder by running the following in the terminal:

```
upzip \*.zip
```

Then, you can import the `preprocess` module to reproduce the `text_data.csv` file and descriptive plots about the text data in the Jupyter Notebook `preprocess.ipynb`. 

## Preliminary Analysis

Finally, you can use the `analyze` module to reproduce the preliminary analysis of the data using Doc2Vec and Word2Vec models in the Jupyter Notebook `analyze.ipynb`. In particular, the `plot_d2v_similarities` function generates a graph of average pairwise cosine similarity score for each year's documents, showing the degree of cohesiveness among the stories Chinese scholars tell about China-Japan relations.

```
data = analyze.prepare_d2v_documents('text_data.csv', 'stopwords-zh.txt')
D2V = analyze.train_best_d2v_model(data)
similarities_lst = analyze.get_d2v_similarities(D2V, data)
analyze.plot_d2v_similarities(similarities_lst, smooth=True)
```

![png](visuals/cohesiveness.png)
