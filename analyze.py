import itertools
import pandas as pd
import numpy as np
import gensim
import re
import preprocess
import nltk
import random
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE 
from sklearn.utils import resample
from scipy.interpolate import make_interp_spline
from pyanno.measures import pairwise_matrix
from pyanno.measures import pearsons_rho, spearmans_rho
from scipy import stats

sns.set(rc={'figure.figsize':(11.7,8.27)}, font_scale=1.3)
sns.set_theme(style="dark")


def prepare_d2v_documents(text_data_filename, stopword_filename):
    """
    Take the 'text_data.csv' file and generate a dataframe ready 
    for training D2V model.
    
    Input:
      text_data_filename: the name of the csv file, i.e. 'text_data.csv')
      stopword_filename: the name of a txt file that stores stopwords
      
    Return a pandas dataframe with a new column 'tagged_docs' ready 
      as input to train Doc2Vec model
    """
    data = pd.read_csv(text_data_filename, index_col=[0])
    data.reset_index(drop=True, inplace=True)
    
    # csv files treat lsts as strs; convert it back
    data['tokenized_text'] = data['tokenized_text'].apply(lambda x: re.split('\W+', x))
    data['tokenized_text'] = data['tokenized_text'].apply(lambda x: x[1:-1])
    
    # remove stopwords
    file = open(stopword_filename, 'r') 
    stopwords = file.read().splitlines()
    file.close()
    data['tokenized_text'] = data['tokenized_text'].apply(lambda x: \
                             [word for word in x if word not in stopwords])
    # select keywords
    keywords = ['中日关系', 
                '军事', '政治', '贸易', '历史', '地缘', '安全',  '文化', 
                '企业', '经济', '全球化', 
                '威胁论', '军国主义', '钓鱼岛', '侵华', '靖国神社', '侵略',
                '客观', '理性', '情感', '共识', '伙伴', '友好', '情绪', '精神',
                '台湾', '美国', '俄', '俄罗斯', '西方', '日美',
                '前景', '新', '发展', '冲突', '争议', '合作']
    
    # tag documents
    taggedDocs = []
    for index, row in data.iterrows():
        docKeywords = [s for s in keywords if s in row['tokenized_text']]
        docKeywords.append(str(index))
        taggedDocs.append(gensim.models.doc2vec.TaggedDocument(words=row['tokenized_text'], 
                                                               tags=docKeywords))
    data['tagged_docs'] = taggedDocs
    
    return data


def get_d2v_validation(validation_filename):
    """
    Take the 'annotation.csv' file and generate a dataframe containing 
    validation data.
    
    Input:
      validation_filename: the name of the csv file, i.e. 'annotation.csv')      
    
    Return a pandas dataframe of validation data
    """
    validation = pd.read_csv(validation_filename, index_col=[0])
    
    # csv files treat lsts as strs; convert it back
    validation['tokenized_text1'] = validation['tokenized_text1'].apply(lambda x: re.split('\W+', x))
    validation['tokenized_text1'] = validation['tokenized_text1'].apply(lambda x: x[1:-1])
    validation['tokenized_text2'] = validation['tokenized_text2'].apply(lambda x: re.split('\W+', x))
    validation['tokenized_text2'] = validation['tokenized_text2'].apply(lambda x: x[1:-1])
    
    return validation

def score_d2v_model(model, validation):
    """
    Calculate the mean squared error between human annotated similarity scores
    and a d2v model's predicted similarity scores, given a set of paired documents
    
    Input:
      validation: a pandas dataframe stored the validation dataset
      model: the d2v model to be tested
      return_pred: whether to return the predicted cosine similarity score 
    
    Return a tuple, the first item being the mean squared error, the second being 
    the predicted cosine similarity score scaled to 0 and 1 using sklearn's MinMaxScaler
    """
    
    sims = []
    for i, row in validation.iterrows():
        v1 = model.infer_vector(row['tokenized_text1']) 
        v2 = model.infer_vector(row['tokenized_text2'])
        sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
        sims.append(sim)
    scaler = MinMaxScaler()
    pred = scaler.fit_transform(np.array(sims).reshape(-1,1)).reshape(1,-1)[0]
    mse = mean_squared_error(validation['similiarity_score'], pred)
    
    return mse, pred


def grid_search(data, key, search_space, validation):
    """
    Find best hyper-parameters for the d2v model based on performance of the model
    on validation data
    
    Input:
      data: the pandas dataframe that contains text data
      key: the str of parameter name
      search_space: a list of possible values to try on
      validation: a pandas dataframe stored the validation dataset
    
    Return a dictionary mapping parameter values to test scores, and a dictionary 
    mapping parameter values to lists of prediction results
    """
    results = {}
    preds = {}
    for v in search_space:
        if key == 'dm':
            D2V = gensim.models.doc2vec.Doc2Vec(dm=v, 
                                          vector_size=300,
                                          window=8,
                                          min_count=3,
                                          workers=7,
                                          dm_concat=1)
        elif key == 'vector_size':
            D2V = gensim.models.doc2vec.Doc2Vec(dm=0, 
                                          vector_size=v,
                                          window=8,
                                          min_count=3,
                                          workers=7,
                                          dm_concat=1)
        elif key == 'window':
            D2V = gensim.models.doc2vec.Doc2Vec(dm=0, 
                                          vector_size=300,
                                          window=v,
                                          min_count=3,
                                          workers=7,
                                          dm_concat=1)
        elif key == 'dm_concat':
            D2V = gensim.models.doc2vec.Doc2Vec(dm=0, 
                                          vector_size=300,
                                          window=12,
                                          min_count=3,
                                          workers=7,
                                          dm_concat=v)    
        
        elif key == 'dm_mean':
            D2V = gensim.models.doc2vec.Doc2Vec(dm=0, 
                                          vector_size=300,
                                          window=12,
                                          min_count=3,
                                          workers=7,
                                          dm_concat=1,
                                          dm_mean=v)
        
        elif key == 'min_count':
            D2V = gensim.models.doc2vec.Doc2Vec(dm=0, 
                                          vector_size=300,
                                          window=12,
                                          min_count=v,
                                          workers=7,
                                          dm_concat=1,
                                          dm_mean=0) 
        
        D2V.build_vocab(data['tagged_docs'])
        D2V.train(data['tagged_docs'], total_examples=D2V.corpus_count, epochs=10)
        
        mse, pred = score_d2v_model(D2V, validation)
        results[v] = mse
        preds[v] = pred
        del D2V
        
    return results, preds


def train_best_d2v_model(data):
    """
    Train a d2v model using the best set of parameters found by gridsearch.
    
    Input:
      data: the pandas dataframe that contains text data
    Return a Gensim d2v object. Also save model as 'D2V' in the current directory.
    """    
    D2V = gensim.models.doc2vec.Doc2Vec(dm=0, 
                                        vector_size=300,
                                        window=12,
                                        min_count=3,
                                        workers=7,
                                        dm_concat=1,
                                        dm_mean=0) 
        
    D2V.build_vocab(data['tagged_docs'])
    D2V.train(data['tagged_docs'], total_examples=D2V.corpus_count, epochs=10)
    D2V.save('D2V')
    
    return D2V


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.dv.vectors)
    labels = np.asarray(model.dv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    
    return x_vals, y_vals, labels


def plot_with_matplotlib(model, data, year1, year2):
    
    x_vals, y_vals, labels = reduce_dimensions(model)
    df = pd.DataFrame({'Reduced Dimension 1': x_vals, 
                       'Reduced Dimension 2': y_vals, 
                       'label': labels})
    year_book = {}
    for i, row in data.iterrows():
        current_year = row['year']
        if current_year not in year_book:
            year_book[current_year] = []
        year_book[current_year].append(str(i))
        
    indices1 = year_book[year1]
    indices2 = year_book[year2]
        
    year = []
    for i, row in df.iterrows():
        if row['label'] in indices1:
            year.append(year1)
        elif row['label'] in indices2:
            year.append(year2)
        else:
            year.append("Other Years")
    df['Year'] = year
        
        
    sns.scatterplot(data=df, x="Reduced Dimension 1", 
                    y="Reduced Dimension 2", 
                    hue='Year',
                    size='Year',
                    sizes={'Other Years': 40, year1: 200, year2: 200},
                    alpha=0.8, 
                    palette="Set2")
    plt.title("T-SNE visualization of article embeddings", fontsize=16)
    plt.xlabel("Reduced Dimension 1", fontsize=16)
    plt.ylabel("Reduced Dimension 2", fontsize=16)
    plt.show()

    
def get_d2v_similarities(model, data, return_vectors=False):
    """
    Calculate the average pairwise document similarities across years, 
    using a d2v model.
    
    Input:
      data: the pandas dataframe that contains text data
      model: the trained d2v model
    Return a list of average similarity scores, ordered by year.
    """
    year_book = {}
    for i, row in data.iterrows():
        current_year = row['year']
        if current_year not in year_book:
            year_book[current_year] = []
        year_book[current_year].append(i)
        
    similarities_lst = []
    vector_dic = {}
    for year in range(2000, 2022):
        total_similarities = 0
        index_lst = year_book[year]
        permutations = list(itertools.combinations(index_lst, 2))
        for pair in permutations:
            try:
                m = model[str(pair[0])]
            except:
                m = model.infer_vector(data['tokenized_text'][pair[0]])
            try:
                n = model[str(pair[1])]
            except:
                n = model.infer_vector(data['tokenized_text'][pair[1]])                
            total_similarities += (cosine_similarity(m.reshape(1,-1), 
                                                        n.reshape(1,-1)))[0][0]
        normalized_similarities = total_similarities / len(permutations)
        similarities_lst.append(normalized_similarities)
        
        if return_vectors:
            for article in index_lst:
                try:
                    vector = normalize(model[str(article)])
                except:
                    vector = normalize(model.infer_vector(data['tokenized_text'][article]))
                vector_dic[article] = vector
        
    if return_vectors:
        return similarities_lst, vector_dic
    
    return similarities_lst
    

def get_within_year_divergence(data):
    """
    Calculate the average pairwise KS divergence of documents across years.
    Input:
      data: the pandas dataframe that contains text data
    Return a list of average pairwise KS divergence scores, ordered by year.
    """
    year_book = {}
    for i, row in data.iterrows():
        current_year = row['year']
        if current_year not in year_book:
            year_book[current_year] = []
        year_book[current_year].append(i)

    divergence_lst = []
    for year in range(2000, 2022):
        # get a dictionary that maps each article to count dictionary
        dic_book = {}
        for article in year_book[year]:   
            distribution = nltk.FreqDist(data['tokenized_text'][article])
            dic_book[article] = pd.DataFrame(list(distribution.values()), 
                                             columns = ['frequency'], 
                                             index = list(distribution.keys()))       
        total_divergence = 0
        index_lst = year_book[year]
        permutations = list(itertools.combinations(index_lst, 2))
        for pair in permutations:
            P = dic_book[pair[0]]
            Q = dic_book[pair[1]]
            P.columns = ['P']
            Q.columns = ['Q']
            df = Q.join(P).fillna(0)
            p = df.iloc[:,1]
            q = df.iloc[:,0]
            divergence = stats.ks_2samp(p, q).statistic        
            total_divergence += divergence
            
        normalized_divergence = total_divergence / len(permutations)
        divergence_lst.append(normalized_divergence)
        
    return divergence_lst


def get_KS_pair_divergence(wordlist1, wordlist2):
    """
    Calculate pairwise KS divergence between two word distributions, 
    given two wordlists.
    Input:
      wordlist1, wordlist2: two lists of tokens to be compared
    Return a float of KS distance between two word distributions
    """
    distribution1 = nltk.FreqDist(wordlist1)
    distribution2 = nltk.FreqDist(wordlist2)
    P = pd.DataFrame(list(distribution1.values()), columns = ['frequency'], 
                     index = list(distribution1.keys())) 
    Q = pd.DataFrame(list(distribution2.values()), columns = ['frequency'], 
                     index = list(distribution2.keys())) 
    P.columns = ['P']
    Q.columns = ['Q']
    df = Q.join(P).fillna(0)
    p = df.iloc[:,1]
    q = df.iloc[:,0]
    divergence = stats.ks_2samp(p, q).statistic  
    
    return divergence


def transform_d2v_similarities(similarities_lst):
    """
    Transform the values to diversity and scale it to 0 and 1. 
    """
    diversity = [1-x for x in similarities_lst]
    scaler = MinMaxScaler()
    transformed_diversity =scaler.fit_transform(\
        np.array(diversity).reshape(-1,1)).reshape(1,-1)[0]
    
    return transformed_diversity


def plot_d2v_similarities(transformed_diversity, smooth=False):
    """
    Plot the average pairwise document similarities across years.
    Input:
      similarities_lst: a list of average similarity scores, ordered by year.
      smooth: a boolean value indicating whether the plot will be smoothed
    """
    
    if not smooth:
        plt.plot(range(2000,2022), transformed_diversity)
        plt.title("Diversity of the Scholarship")
        plt.xlabel("Year")
        plt.ylabel("Diversity")
        plt.show()       
    
    else:
        x = np.array(range(2000,2022))
        y = np.array(transformed_diversity)
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)  
        sns.lineplot(data=pd.DataFrame({'Year': X_,
                                        'Average Pairwise Cosine Similarity': Y_}), 
                                        x="Year", 
                                        y="Average Pairwise Cosine Similarity")
        plt.title("""Diversity of the Scholarship""", fontsize=16)
        plt.axvspan(2001, 2006, -1,+1, facecolor='peachpuff')
        plt.axvspan(2010, 2014, -1,+1, facecolor='peachpuff', label = 'Contentious Years')
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("Diversity", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
                         

def plot_three_measures(similarities_lst, divergence_lst, perplexity, \
                        transform_only=False):
    """
    Transform three measures of diversity within the same scale 
    from 0 to 1, and plot them in the same graph.
    """
    scaler1 = MinMaxScaler()
    transformed_divergence =scaler1.fit_transform(\
        np.array(divergence_lst).reshape(-1,1)).reshape(1,-1)[0]
    
    scaler2 = MinMaxScaler()
    transformed_similarity = scaler2.fit_transform(\
    np.array(similarities_lst).reshape(-1,1)).reshape(1,-1)[0]
    
    scaler3 = MinMaxScaler()
    perplexity = np.array(perplexity)
    perplexity = 1 - np.exp(perplexity - perplexity.max(axis=0))
    transformed_perplexity = scaler3.fit_transform(\
        np.array(perplexity).reshape(-1,1)).reshape(1,-1)[0]
        
    x = np.array(range(2000,2022))
    three_measures = pd.DataFrame({'Year': x,
    'D2V Model': transformed_similarity,
    'KS Divergence Model': transformed_divergence,
    'Perplexity Model': transformed_perplexity})

    if transform_only:
        return three_measures
    
    three_measures = three_measures.melt(id_vars=["Year"], 
                                         var_name="Model", value_name="Measure")
    sns.color_palette("Set2")
    sns.lineplot(data=three_measures, 
                 x='Year', 
                 y='Measure', 
                 hue='Model',
                 dashes=False)
    plt.title("""Comparing Three Measures of Intellectual Diversity""", fontsize=16)
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Degree of Diversity by Each Measure", fontsize=16)
    plt.axvspan(2001, 2006, -1,+1, facecolor='peachpuff')
    plt.axvspan(2010, 2014, -1,+1, facecolor='peachpuff', label = 'Contentious Years')
    plt.show()    

    
def compare_three_measures(similarities_lst, divergence_lst, perplexity,visual=True):
    """
    Calculate Pearson correlation of three measures of cohesiveness, 
    and visualize it in heatmap.
    """
    three_measures = plot_three_measures(\
                                similarities_lst, divergence_lst, perplexity, \
                                transform_only=True)
    three_measures.drop(columns=['Year'], inplace=True)
    
    n = pairwise_matrix(pearsons_rho, np.array(three_measures))
    if not visual:
        return n
    else:
        labels = ['D2V','Divergence','Perplexity']
        sns.heatmap(n, xticklabels = labels, yticklabels = labels, 
                    cmap="YlGnBu", annot=True)
        plt.title("""Pearson Correlation between Three Measures of Intellectual Diversity""", fontsize=16)
        plt.show()           
        
        
def train_w2v(data):
    """
    Train w2v models on every year's articles.
    Models will be saved as'w2v_(year)', eg. w2v_2000.
    """
    for year in range(2000, 2022):
        sentences = data[data['year'] == year]['tokenized_text']
        model = Word2Vec(sentences=sentences,
                         vector_size=300, 
                         window=5, 
                         min_count=3, 
                         workers=7,
                         sg=1,
                         epochs=10)  

        model.save('w2v_{}'.format(year))
        

def train_w2v_for_perplexity(data):
    """
    Train w2v models on every year's articles.
    Models will be saved as'perp_w2v_(year)', eg. w2v_2000.
    """
    for year in range(2000, 2022):
        sentences = data[data['year'] == year]['tokenized_text']
        model = Word2Vec(sentences=sentences,
                         vector_size=300, 
                         window=5, 
                         min_count=3, 
                         workers=7,
                         sg=1,
                         epochs=5)  

        model.save('perp_w2v_{}'.format(year))

        
def get_w2v_perplexity(data, current=True):
    """
    Given a w2v model trained on each year's texts, measure the perplexity 
    of the model when encountering each text from this year. 
    
    Input:
      data: the pandas dataframe that contains text data
      current: If current=False, using the w2v model trained on previous 
        year's text.    
    Return a list of average perplexity ordered by year.
    """
    perplexity_lst = []
    year_book = {}
    for i, row in data.iterrows():
        current_year = row['year']
        if current_year not in year_book:
            year_book[current_year] = []
        year_book[current_year].append(i)
    
    for year in range(2000, 2022):
        model_name = 'perp_w2v_{}'.format(year) 
        model = Word2Vec.load('perp_w2v/' + model_name)        
        
        total_perplexity = 0    
        for article in year_book[year]:
            perplexity = model.score(data['tokenized_text'][article],
                               total_sentences=1, 
                               chunksize=100) 
            total_perplexity += perplexity
        average_perplexity = total_perplexity / len(year_book[year])       
        perplexity_lst.append(average_perplexity[0])
        
    return perplexity_lst
            

def normalize(vector):
    '''
    A helper function to normalize the word vector
    '''
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector
           

def project_word(keyword, neg_file, pos_file):
    """
    Project a keyword along the dimensions of positive and negative sentiment
    across years, using each year's w2v model.
    Input:
      keyword: the word to be projected
      neg_file: the text file name that contains negative words in Chinese 
      pos_file: the text file name that contains positive words in Chinese 
    Return a list of normalized position of the word on the dimension
        across years
    """
    # read positive and negative words dictionary 
    # sources: https://github.com/bung87/bixin/blob/master/data/neg.txt
    
    file = open(neg_file, 'r') 
    neg = file.read().splitlines()
    file.close()

    file = open(pos_file, 'r') 
    pos = file.read().splitlines()
    file.close()
    
    # load models
    projection = []
    for year in range(2000, 2022):
        model_name = 'w2v_{}'.format(year)
        model = Word2Vec.load('w2v/' + model_name)

        # get negative/positive words appear in model vocab
        neg_appear = []
        pos_appear = []
        for word in model.wv.index_to_key:
            if word in neg:
                neg_appear.append(word)
            elif word in pos:
                pos_appear.append(word)

        # construct dimension
        p_vec = []
        n_vec = []
        for word in neg_appear:
            n_vec.append(normalize(model.wv[word]))
        for word in pos_appear:
            p_vec.append(normalize(model.wv[word]))
        diff = sum(p_vec) - sum(n_vec)

        projection.append(cosine_similarity(normalize(model.wv[keyword]).reshape(1,-1),
                                        diff.reshape((1,-1)))[0][0])

    return projection


def plot_projection(keyword, dimension, projection):
    """
    Plot the projection of a keyword on a dimension across years.
    
    Input:
      keyword: the English-translated keyword that was projected
      dimension: the name of the dimension, e.g. emotion vs. rational
      projection: a list of normalized position of the word on the dimension
        across years
    """
    x = np.array(range(2000,2022))
    y = np.array(projection)
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    sns.lineplot(data=pd.DataFrame({'Year': X_,
                                    'Sentiment': Y_}), 
                                        x="Year", 
                                        y="Sentiment",
                                        linewidth = 2)
    plt.title("Texture of bilateral relations as perceived by scholars", fontsize=16)
    plt.axvspan(2001, 2006, -1,+1, facecolor='peachpuff', label = 'Contentious Years')
    plt.axvspan(2010, 2014, -1,+1, facecolor='peachpuff')
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Negative vs. Positive", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.show();    
    
    
def run_x_y_regression(y, y_name, x, x_name, plot):
    """
    Run an ols model testing the effect of X on Y.
    """    
    cor = pd.DataFrame({'{}'.format(x_name): x, 
                        '{}'.format(y_name): y})
    y = cor[y_name]
    x = cor[x_name]
    x = sm.add_constant(x)
    ols_model = sm.OLS(y, x)  
    results = ols_model.fit()
    
    if plot:
        sns.lmplot(x=x_name, y=y_name, data=cor,
                    truncate=False, robust=True)
        plt.title("The effect of {} on {}".format(x_name, y_name), fontsize=16)
        plt.xlabel(x_name, fontsize=16)
        plt.ylabel(y_name, fontsize=16)
        plt.show();
    
    
    return results