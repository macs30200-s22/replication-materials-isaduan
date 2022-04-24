# Chinese Elite Nationalism: Supplemental Code

The code and data in this repository is an example of a reproducible research workflow for the research project "Reading Chinese Elite Nationalism from Foreign Policy Scholarship on China-Japan Relations." The code is written in Python 3.7.13 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

Next, you can download the `raw_data` folder and unzip all files wihin the folder by running the following in the terminal:

```
upzip \*.zip
```

Then, you can import the `preprocess` module located in this repository to reproduce the `text_data.csv` file and descriptive plots about the text data in the Jupyter Notebook `preprocess.ipynb`. 

Alternatively, to replicate the analysis and produce all of the figures and quantitative analyses from the (hypothetical) publication that this code supplements, build and run the `Dockerfile` included in this repository via the instructions in the file).

If you use this repository for a scientific publication, we would appreciate it if you cited the [Zenodo DOI](https://doi.org/10.5281/zenodo.6429151) (see the "Cite as" section on our Zenodo page for more details).
