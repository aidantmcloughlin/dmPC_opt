import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math

#-------------------------------------------------------------------------------------------------------------------------------------------
# CDR relevent functions
def binarize_CDR(CDR, method="cutoff", cutoff=3.5):
    """
    CDR: a DataFrame , cancer drug response values,
         with rows for cancer cell lines and columns for drugs
    method: a string, binarization methof, either "cutoff" or "waterfall"
    cutoff: a number, cutoff value if method == "cutoff"
    """
    if method == 'waterfall':
        binarized_CDR = pd.DataFrame(np.apply_along_axis(binarize_CDR_waterfall, axis=1, arr=CDR))
    else: 
        binarize = lambda x: 1 if x <= cutoff else 0
        binarized_CDR = CDR.applymap(binarize)

    return binarized_CDR


def binarize_CDR_waterfall(CDR):
    """
    CDR1: a 1D array of CDR values from one drug
    """
    # 1. sorts cell lines according to their AUC values in descending order
    CDR_sorted = np.sort(CDR)[::-1]
    orders = np.arange(len(CDR_sorted), 0, -1)
    
    # 2. generates an AUC-cell line curve in which the x-axis represents cell lines and the y-axis represents AUC values
    
    # 3. generate the cutoff of AUC values
    cor_pearson, _ = pearsonr(orders, CDR_sorted)
    
    if cor_pearson > 0.95:
        # 3.1. for linear curves (whose regression line fitting has a Pearson correlation >0.95), 
        #      the sensitive/resistant cutoff of AUC values is the median among all cell lines
        cutoff = np.median(CDR)
    else:
        # 3.2 otherwise, the cut off is the AUC value of a specific boundary data point. 
        #     It has the largest distance to a line linking two data points having the largest and smallest AUC values
        cutoff = max_dist_MinMaxLine(CDR_sorted)
    
    binarized_CDR = np.zeros_like(CDR)
    binarized_CDR[CDR < cutoff] = 1
    binarized_CDR[CDR >= cutoff] = 0
    return binarized_CDR


def max_dist_MinMaxLine(points):
    # Helper function to find the maximum distance between a point and a line linking two other points
    x1, y1 = 1, points[0]
    x2, y2 = len(points), points[-1]
    dists = [np.abs((y2-y1)*x + (x1-x2)*y + (x2*y1 - x1*y2)) / np.sqrt((y2-y1)**2 + (x1-x2)**2) for x, y in enumerate(points)]
    return points[np.argmax(dists)]


#-------------------------------------------------------------------------------------------------------------------------------------------
def one_hot_encode(string_list):

    unique_strings = list(set(string_list))
    string_to_index = {string: index for index, string in enumerate(unique_strings)}

    encoded_strings = []
    for string in string_list:
        one_hot = string_to_index[string]
        encoded_strings.append(one_hot)

    return encoded_strings, string_to_index


#-------------------------------------------------------------------------------------------------------------------------------------------
# Cancer cell line meta functions
def get_CCL_meta_codes(CCL_names, meta):
    columns = pd.DataFrame({'C_ID': CCL_names})
    columns['C_ID'] = columns['C_ID'].astype(str)
    
    meta['C_ID'] = meta['C_ID'].astype(str)
    
    meta = pd.merge(columns, meta, on=['C_ID'], how='left')
    meta = meta.set_index('C_ID', drop=True).rename_axis(None)
    
    meta.C_type = pd.Categorical(meta.C_type)
    meta['code'] = meta.C_type.cat.codes
    
    # meta_map = meta[['C_type', 'code']].value_counts().index.values
    meta_map = meta.groupby(['C_type', 'code']).size()
    meta_map = meta_map.reset_index(name='count')
    meta_map = meta_map.loc[meta_map['count'] != 0]
        
    meta = meta.drop("C_type", axis=1)

    for k in meta['code'].unique():
        meta[f'k{k}'] = (meta.code==k).astype(int)
    
    return(meta, meta_map)


def get_CCL_meta(CCL_names, meta):
    columns = pd.DataFrame({'C_ID': CCL_names})
    columns['C_ID'] = columns['C_ID'].astype(str)
    
    meta['C_ID'] = meta.index.values.astype(str)
    
    meta = pd.merge(columns, meta, on=['C_ID'], how='left')
    meta = meta.set_index('C_ID', drop=True).rename_axis(None)
    
    return(meta)

#-------------------------------------------------------------------------------------------------------------------------------------------
# Get outlier index functions
def find_outliers_IQR(dists):
    
    q1=dists.quantile(0.25)
    q3=dists.quantile(0.75)
    IQR=q3-q1
    
    outliers_idx = ((dists<(q1-1.5*IQR)) | (dists>(q3+1.5*IQR)))
    outliers_idx = outliers_idx.flatten().numpy()
    outliers_idx = np.where(outliers_idx)
    return outliers_idx

def find_outliers_3sd(latent):
    # upper_limit = latent.mean() + 3*latent.std()
    # lower_limit = latent.mean() - 3*latent.std()
    
    # outliers_idx = (latent > upper_limit)|(latent < lower_limit)
    # outliers_idx = outliers_idx.flatten().numpy()
    # outliers_idx = np.where(outliers_idx)
    # return outliers_idx

    mean_tensor = latent.mean(dim=0)
    sd_tensor = latent.std(dim=0)

    # Define the threshold for outliers as 3 times the standard deviation
    threshold = 3 * sd_tensor

    # Find outliers by comparing each tensor to the threshold
    outliers_bool = (latent - mean_tensor).abs() > threshold
    index_of_outlier = outliers_bool.any(dim=1).nonzero().view(-1).numpy()

    return index_of_outlier




#-------------------------------------------------------------------------------------------------------------------------------------------
def get_C_sensitive_codes(cdr_k, sens_cutoff):
    """
    cdr_k: a DataFrame , cancer drug response values,
            with three columns: c_name, d_name, cdr
    return cdr_k_avg: a data frame with one binary column, 
                      indicating if the cancer cell line is sensitive to the drug group in the cdr_k.
                      indexed by d_name
    """
    cdr_k_avg = cdr_k.groupby(['c_name'])['cdr'].mean().reset_index()
    cdr_k_avg = cdr_k_avg.rename(columns={'cdr':'avg_cdr'})

    cdr_k_avg['sensitive'] = (cdr_k_avg.avg_cdr > sens_cutoff).astype(int)
    
    cdr_k_avg = cdr_k_avg.set_index('c_name', drop=True).rename_axis(None)
    cdr_k_avg= cdr_k_avg.drop("avg_cdr", axis=1)
    
    return(cdr_k_avg)
    
    
def get_D_sensitive_codes(cdr_k, sens_cutoff):
    """
    cdr_k: a DataFrame , cancer drug response values,
            with three columns: c_name, d_name, cdr
    return cdr_k_avg: a data frame with one binary column, 
                      indicating if the drug is sensitive to the cancer cell line group in the cdr_k.
                      indexed by d_name
    """
    cdr_k_avg = cdr_k.groupby(['d_name'])['cdr'].mean().reset_index()
    cdr_k_avg = cdr_k_avg.rename(columns={'cdr':'avg_cdr'})

    cdr_k_avg['sensitive'] = (cdr_k_avg.avg_cdr > sens_cutoff).astype(int)
    
    cdr_k_avg = cdr_k_avg.set_index('d_name', drop=True).rename_axis(None)
    cdr_k_avg= cdr_k_avg.drop("avg_cdr", axis=1)
    
    return(cdr_k_avg)

def get_D_sensitive(d_names, d_sens_cluster_k):
    columns = pd.DataFrame({'d_name': d_names})
    columns['d_name'] = columns['d_name'].astype(str)
    
    d_sens_cluster_k['d_name'] = d_sens_cluster_k.index.values.astype(str)
    
    d_sens_cluster_k = pd.merge(columns, d_sens_cluster_k, on=['d_name'], how='left')
    d_sens_cluster_k = d_sens_cluster_k.set_index('d_name', drop=True).rename_axis(None)
    
    return(d_sens_cluster_k)
#-------------------------------------------------------------------------------------------------------------------------------------------
# Get training loss history
def get_train_VAE_hist_df(train_hist, n_epochs):
    losses = {'epoch': range(n_epochs),
          'loss_train': [value for key, value in train_hist[0].items() if key[1] == 'train'],
          'loss_test': [value for key, value in train_hist[0].items() if key[1] == 'val'],
          'recon_loss_train': [value for key, value in train_hist[1].items() if key[1] == 'train'],
          'recon_loss_test':[value for key, value in train_hist[1].items() if key[1] == 'val'],
          'kld_train': [value for key, value in train_hist[2].items() if key[1] == 'train'],
          'kld_test': [value for key, value in train_hist[2].items() if key[1] == 'val']
         }
  
    losses = pd.DataFrame(losses)
    return(losses)


def get_train_VAE_predictor_hist_df(train_hist, n_epochs, vae_type = "C"):
    if vae_type == "C":
        losses = {'epoch': range(n_epochs),
          'loss_train': [value for key, value in train_hist[0].items() if key[1] == 'train'],
          'loss_test': [value for key, value in train_hist[0].items() if key[1] == 'val'],
          'vae_loss_train': [value for key, value in train_hist[1].items() if key[1] == 'train'],
          'vae_loss_test': [value for key, value in train_hist[1].items() if key[1] == 'val'],
          'recon_loss_train': [value for key, value in train_hist[2].items() if key[1] == 'train'],
          'recon_loss_test':[value for key, value in train_hist[2].items() if key[1] == 'val'],
          'kld_train': [value for key, value in train_hist[3].items() if key[1] == 'train'],
          'kld_test': [value for key, value in train_hist[3].items() if key[1] == 'val'],
          'latent_dist_loss_train': [value for key, value in train_hist[4].items() if key[1] == 'train'],
          'latent_dist_loss_test': [value for key, value in train_hist[4].items() if key[1] == 'val'],
          'update_overlap_train': [value for key, value in train_hist[5].items() if key[1] == 'train'],
          'update_overlap_test': [value for key, value in train_hist[5].items() if key[1] == 'val'],
          'prediction_loss_train': [value for key, value in train_hist[11].items() if key[1] == 'train'],
          'prediction_loss_test': [value for key, value in train_hist[11].items() if key[1] == 'val']
         }
    if vae_type == "D":
        losses = {'epoch': range(n_epochs),
          'loss_train': [value for key, value in train_hist[0].items() if key[1] == 'train'],
          'loss_test': [value for key, value in train_hist[0].items() if key[1] == 'val'],
          'vae_loss_train': [value for key, value in train_hist[6].items() if key[1] == 'train'],
          'vae_loss_test': [value for key, value in train_hist[6].items() if key[1] == 'val'],
          'recon_loss_train': [value for key, value in train_hist[7].items() if key[1] == 'train'],
          'recon_loss_test':[value for key, value in train_hist[7].items() if key[1] == 'val'],
          'kld_train': [value for key, value in train_hist[8].items() if key[1] == 'train'],
          'kld_test': [value for key, value in train_hist[8].items() if key[1] == 'val'],
          'latent_dist_loss_train': [value for key, value in train_hist[9].items() if key[1] == 'train'],
          'latent_dist_loss_test': [ value for key, value in train_hist[9].items() if key[1] == 'val'],
          'update_overlap_train': [value for key, value in train_hist[10].items() if key[1] == 'train'],
          'update_overlap_test': [value for key, value in train_hist[10].items() if key[1] == 'val'],
          'prediction_loss_train': [ value for key, value in train_hist[11].items() if key[1] == 'train'],
          'prediction_loss_test': [value for key, value in train_hist[11].items() if key[1] == 'val']
         }
    losses = pd.DataFrame(losses)
    return(losses)


#-------------------------------------------------------------------------------------------------------------------------------------------

def add_meta_code(df: np.ndarray, K: int, B: int) -> pd.DataFrame:
    df = pd.DataFrame(df)
    
    for b in range(B):
        code_column = pd.Series("", index=df.index)
    
        for k in range(K):
            col_name = f'k{k}_b{b}'
            code_column[df[col_name] == 1] += str(k) + ' & '
            
        code_column = code_column.str.rstrip(' & ')
        code_column = code_column.replace('', '-1')
    
        df['code_b' + str(b)] = code_column
    
    return df

def add_meta_code_with_subcluster(df: np.ndarray, K: int, B: int) -> pd.DataFrame:
    df = pd.DataFrame(df)
    
    for b in range(B):
        code_column = pd.Series("", index=df.index)
    
        for k in range(K):
            col_name = f'k{k}_b{b}'
            code_column[df[col_name] == 1] += str(k) + ' & '
            
        code_column = code_column.str.rstrip(' & ')
        code_column = code_column.replace('', '-1')
    
        df['code_sub_b' + str(b)] = code_column
    
    return df

def add_sensk_to_d_sens_init(df: np.ndarray, K: int) -> pd.DataFrame:
    df = pd.DataFrame(df)

    code_column = pd.Series("", index=df.index)
    
    b = -1
    for k in range(K):
        col_name = f'sensitive_k{k}'
            
        code_column[df[col_name] == 1] += str(k) + ' & '
            
    code_column = code_column.str.rstrip(' & ')
    code_column = code_column.replace('', '-1')
        
    df['sensitive_k'] = code_column

    return df

def add_sensk_to_d_sens_hist(df: np.ndarray, K: int, B: int) -> pd.DataFrame:
    df = pd.DataFrame(df)
    
    for b in range(0, B):
        code_column = pd.Series("", index=df.index)
        
        for k in range(K):
            col_name = f'sensitive_k{k}_b{b}'
            
            code_column[df[col_name] == 1] += str(k) + ' & '
            
        code_column = code_column.str.rstrip(' & ')
        code_column = code_column.replace('', '-1')
        
        df['sensitive_k_b' + str(b)] = code_column

    return df

def add_sensk_to_d_sens_hist_with_subcluster(df: np.ndarray, K: int, B: int) -> pd.DataFrame:
    df = pd.DataFrame(df)
    
    for b in range(0, B):
        code_column = pd.Series("", index=df.index)
        
        for k in range(K):
            col_name = f'sensitive_k{k}_b{b}'
            
            code_column[df[col_name] == 1] += str(k) + ' & '
            
        code_column = code_column.str.rstrip(' & ')
        code_column = code_column.replace('', '-1')
        
        df['sensitive_k_sub_b' + str(b)] = code_column

    return df


def format_list_as_string(lst):
    if len(lst) == 1:
        return str(lst[0])
    else:
        return " & ".join(map(str, lst))




