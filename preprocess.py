import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import jieba
jieba.load_userdict('jieba_dictionary.txt')
import tika
tika.initVM()
from tika import parser

sns.set(rc={'figure.figsize':(11.7,16)}, font_scale=1.3)
sns.set_theme(style="dark")


def get_dataframe(folder):
    """
    Extract data from a folder of articles stored in pdf files, including each
    article's 'title', 'author', 'filepath', 'year'; also read raw texts from 
    pdf files and tokenize the raw texts.

    Input:
    folder: the name of folder, e.g. '2020'

    Return two items, the first being a pandas dataframe with column 'title', 
    'author', 'filepath', 'year', 'raw_text', 'tokenized_text'; the second being
    a tuple, where the first integer is total number of articles downloaded, the
    second integer is the total number of articles successfully processed
    """
    # extract article meta-data
    paths = []
    authors = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            try:
                names.append(re.findall(r'[\u4e00-\u9fff]+_', filename)[0][:-1])
            except:
                names.append(filename[:-4])
            paths.append(os.path.abspath(folder) + '/' + filename)
            try:
                authors.append(re.findall(r'_[\u4e00-\u9fff]+\.pdf', filename)[0][1:-4])
            except:
                authors.append('NA')
    years = [folder for i in range(len(names))]
    data = pd.DataFrame({'title': names, 'author': authors, 'filepath': paths, 'year': years })
    prev_total = len(data)

    # read raw texts
    contents = []
    for path in data.filepath:
        parsed = parser.from_file(path)
        contents.append(parsed['content'])
    data['raw_text'] = contents
    data['normalized_text'] = data['raw_text'].apply(lambda x: getChinese(x))

    # drop rows that have reading errors
    error_row_index = []
    for i, row in data.iterrows():
        if len(row['normalized_text']) <= 100:
            error_row_index.append(i)

    data.drop(index=error_row_index, inplace=True)

    # tokenize 
    data['tokenized_text'] = data['normalized_text'].apply(lambda x: jieba_tokenizer(x))
    data['length'] = data['tokenized_text'].apply(lambda x: len(x))

    post_total = len(data)
    stats = prev_total, post_total

    return data, stats


def getChinese(context):
    """
    A helper function to extract only Chinese characters in a string using 
    regular expression.
    """
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    try:
        context = filtrate.sub(r'', context) # remove all non-Chinese characters     
    except:
        return ''
        
    return context


def jieba_tokenizer(x):
    """
    A helper function to tokenize a string into a list of strings using jieba,
    a Chinese language tokenizer.
    """
    seg_list = jieba.cut(x, cut_all=False)
    return ",".join(seg_list).split(',')


def plot_descriptive_stats(data, prev_total, post_total, graph_num):
    """
    Generate three descriptive plots of the data:
      [1] Average length of articles 
      [2] Total length of articles 
      [3] Number of articles collected vs. successfully processed 
    """
    if graph_num == 1:
        data.groupby('year')['length'].mean().plot()
        plt.title('Average length of article');
    
    if graph_num == 2:
        data.groupby('year')['length'].sum().plot()
        plt.title('Total length of article');
    
    if graph_num == 3:
        missing_ratio = pd.DataFrame({'collected': prev_total, 'successfully processed': post_total, 
                                      'year': range(2000,2022)})
        missing_ratio.plot(x='year', y=['collected', 'successfully processed'], 
                           kind='bar');
        plt.title('Number of articles collected vs. successfully processed');
    

def plot_sources_or_authors(filename, kind):
    """
    Plot the distribution of the sources or authors of journal articles.
    file
    Input:
      filename: the file that stores the meta-data of articles, i.e. 'meta.txt'
      kind: a string, either 'sources' or 'authors'
    """
    # extract sources and authors
    meta = pd.read_csv(filename) 
    authors = []
    sources =[]
    for i, row in meta.iterrows():
        if re.match('Author-作者.+', row[0]):
            authors.append(row[0])
        if re.match('Source-文献来源.+', row[0]):
            sources.append(row[0])
                
    if kind == 'sources':
        new_sources = []
        for x in sources:
            new_sources.append(x[12:].strip())   
        sources_df = pd.DataFrame({'source': new_sources})
        sns.displot(sources_df.value_counts(), kind="kde", fill=True, \
                    clip=(0, None));
        plt.xlabel("Sources", fontsize=16)
    
    if kind == 'authors':
        new_authors = []
        for x in authors:
            lst = x[12:-1].strip().split(';')
            for author in lst:
                new_authors.append(author)
        authors_df = pd.DataFrame({'source': new_authors})
        sns.displot(authors_df.value_counts(), kind="kde", fill=True, \
                    clip=(0, None));
        plt.xlabel("Authors", fontsize=16)
    
    plt.title("Density distribution of the {} of the articles".format(kind), fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.show();

    
    

    if __name__ == "__main__":
        # When running script in Docker Container, first pre-process the data, 
        # save it as a csv file; then produce descriptive plots
        for year in range(2000, 2022):
            if year == 2000:
                data, stats = get_dataframe('raw_data/' + str(year)) 
                prev_total = [stats[0]]
                post_total = [stats[1]] 
            else:
                new_data, stats = get_dataframe(str(year)) 
                data = pd.concat([data, new_data])
                prev_total.append(stats[0])
                post_total.append(stats[1])

        data.to_csv('text_data.csv')
        for num in range(1, 4):
            plot_descriptive_stats(data, prev_total, post_total, num)
            

