import pickle
import pandas as pd
import numpy as np

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

    # X = add_d2v_columns(X, 'd2v_1_10', data_type=data_type, data_folder='none')

    # X = add_fz_column(X, 'size_diff', data_type=data_type, data_folder=data_folder)
    #
    # X = add_fz_column(X, 'ratio',  data_type=data_type, data_folder=data_folder)
    #
    # X = add_fz_column(X, 'partial_ratio',  data_type=data_type, data_folder=data_folder)
    #
    # X = add_fz_column(X, 'token_sort_ratio',  data_type=data_type, data_folder=data_folder)
    #
    # X = add_fz_column(X, 'token_set_ratio',  data_type=data_type, data_folder=data_folder)
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
