import itertools
import pandas as pd
import numpy as np
import gensim
import re
import preprocess
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import make_interp_spline


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


def get_d2v_similarities(model, data):
    """
    Calculate the average pairwise document similarities across years, using a d2v model.
    
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
    for year in range(2000, 2022):
        total_similarities = 0
        index_lst = year_book[year]
        permutations = list(itertools.combinations(index_lst, 2))
        for pair in permutations:
            total_similarities += (cosine_similarity(model[str(pair[0])].reshape(1,-1), 
                                                        model[str(pair[1])].reshape(1,-1)))[0][0]
        normalized_similarities = total_similarities / len(year_book[year])
        similarities_lst.append(normalized_similarities)
        
    return similarities_lst


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
    

def plot_d2v_similarities(similarities_lst, smooth=False):
    """
    Plot the average pairwise document similarities across years.
    
    Input:
      similarities_lst: a list of average similarity scores, ordered by year.
      smooth: a boolean value indicating whether the plot will be smoothed
    """
    if not smooth:
        plt.plot(range(2000,2022), similarities_lst)
        plt.title("Average Pairwise Similarity between Scholarly Ariticles")
        plt.xlabel("year")
        plt.ylabel("similarity")
        plt.show()       
    
    else:
        x = np.array(range(2000,2022))
        y = np.array(similarities_lst)
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)
        plt.plot(X_, Y_)
        plt.title("Smooth Curve of Average Pairwise Similarity between Scholarly Ariticles")
        plt.xlabel("year")
        plt.ylabel("similarity")
        plt.show()

        
def train_w2v():
    """
    Train w2v models on every year's articles.
    Models will be saved as'w2v_(year)', eg. w2v_2000.
    """
    for year in range(2000, 2022):
        sentences = data[data['year'] == year]['tokenized_text']
        model = Word2Vec(sentences=sentences,
                         vector_size=300, 
                         window=12, 
                         min_count=3, 
                         workers=7,
                         sg=1,
                         epochs=10)  

        model.save('w2v_{}'.format(year))
    

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
    
    file = open('neg.txt', 'r') 
    neg = file.read().splitlines()
    file.close()

    file = open('pos.txt', 'r') 
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

        projection.append(cosine_similarity(model.wv[keyword].reshape(1,-1),
                                        diff.reshape((1,-1)))[0][0])

    return projection

def plot_projection(keyword, projection):
    """
    Plot the projection of a keyword on a dimension across years.
    
    Input:
      keyword: the English-translated keyword that was projected
      projection: a list of normalized position of the word on the dimension
        across years
    """
    x = np.array(range(2000,2022))
    y = np.array(projection)
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_)
    plt.title("Sentiment towards {}".format(keyword))
    plt.xlabel("year")
    plt.ylabel("projection")
    plt.show();