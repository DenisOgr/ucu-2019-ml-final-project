import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

def build_x(X, data_type, data_folder):

    """
    This method build x (added extra features)
    :param env:
    :return:
    """
    X = add_column(X, 'chebyshev', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'braycurtis', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'cosine', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'correlation', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'canberra', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'hausdorff', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'cityblock', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'euclidean', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'l1', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'l2', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'manhattan', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'minkowski', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'sqeuclidean', data_type=data_type, data_folder=data_folder)

    #graph
    X = add_column(X, 'clique_size', data_type=data_type, data_folder=data_folder)

    #fuzzy and distances
    X = add_column(X, 'cosine_distance', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'cityblock_distance', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'jaccard_distance', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'canberra_distance', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'euclidean_distance', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'minkowski_distance', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'braycurtis_distance', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'len_diff', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'token_sort_ratio', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'token_ratio', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'intersection_ratio', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'token_set_ratio', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'partial_ratio', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'n_capital_letters_diff', data_type=data_type, data_folder=data_folder)

    X = add_column(X, 'n_question_marks_diff', data_type=data_type, data_folder=data_folder)


    return X.drop(columns=['question1', 'question2', 'qid1', 'qid2']).dropna()

def pickle_and_remove(obj, file, data_folder):
    """
    This method serialize and store object
    :param obj:
    :param file:
    :param data_folder:
    :return:
    """
    pickle.dump(obj, open(data_folder + file + '.p', 'wb'))
    del obj

def add_column(df, column, data_type, data_folder):
    col_arr = pickle.load(open(data_folder+column+'_'+data_type+'_w.p', 'rb'))
    return pd.concat([df,
                     pd.Series(col_arr, name=column,index=df.index)
                      ], axis=1)

# add features from set 3 (word2vec vectors)
def add_d2v_columns(df, d2v, data_type, data_folder, red_type='umap'):
    if red_type in ['svd','umap']:
        if red_type == 'svd':
            file = d2v+'_'+data_type+'_svd_red.p'
        else:
            file = d2v+'_'+data_type+'_red.p'
        col_arr = pickle.load(open(data_folder+file, 'rb'))
        return pd.concat([df,
                         pd.DataFrame(col_arr, columns=[d2v+'_'+str(i) for i in range(col_arr.shape[1])],index=df.index)
                          ], axis=1)
    else:
        file1 = 'weighted_mean1_'+data_type+'.p'
        col_arr1 = pickle.load(open(data_folder+file1, 'rb'))
        file2 = 'weighted_mean2_'+data_type+'.p'
        col_arr2 = pickle.load(open(data_folder+file2, 'rb'))
        col_arr = np.hstack((np.concatenate([x.reshape(1,-1) for x in col_arr1]),
                   np.concatenate([x.reshape(1,-1) for x in col_arr2])))
        return pd.concat([df,
                     pd.DataFrame(col_arr, columns=[d2v+'_'+str(i) for i in range(col_arr.shape[1])],index=df.index)
                      ], axis=1)

# add features from set 2 (fuzzy metrics)
def add_fz_column(df, column, train_or_test, data_folder):
    col_arr = pickle.load(open(data_folder+column+'_'+train_or_test+'.p', 'rb'))
    return pd.concat([df,
                     pd.Series(col_arr, name=column,index=df.index)
                      ], axis=1)

def plot_roc(fpr, tpr, roc_auc):
    df_real = pd.DataFrame(fpr, columns=['false positive rate'])
    df_real['true positive rate'] = pd.Series(tpr)
    df_real['curve'] = 'model'
    fpr_ideal = np.insert(fpr, 1, 0.00001)
    df_ideal = pd.DataFrame(fpr_ideal, columns=['false positive rate'])
    df_ideal['true positive rate'] = 1.0
    df_ideal['true positive rate'][0] = 0.0
    df_ideal['curve'] = 'ideal'
    df_worst = pd.DataFrame(fpr, columns=['false positive rate'])
    df_worst['true positive rate'] = pd.Series(fpr)
    df_worst['curve'] = 'random guess'
    df = pd.concat([df_real, df_ideal, df_worst])
    pal = {'model': "#3498db", 'random guess':"#e74c3c", 'ideal':"#34495e"}
    ax = sns.relplot('false positive rate', 'true positive rate', hue='curve', data=df,
                linewidth=2.0, palette=pal, kind="line", legend='full', height=5, aspect=7/5)
    ax.set(xlim=(-.05, 1.0), ylim=(0.0, 1.05), title='Receiver operating characteristic\n(area under curve = %0.2f)' % roc_auc)
    return


def bar_plot_maker(data, value_col, name_col, label, title, logscale=False, xticks=None, xticklabels=None):
    f, ax = plt.subplots(figsize=(7, 10))
    # Plot variances
    sns.set_color_codes("pastel")
    sns.barplot(x=value_col, y=name_col, data=data,
                label=label, color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=1, loc="lower right", frameon=True)
    ax.set(ylabel="", title=title)
    if logscale:
        ax.set(xscale='log')
    if xticks:
        ax.set(xticks=xticks, xticklabels=xticklabels)
    if logscale:
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    sns.despine(left=True, bottom=True)
    return



