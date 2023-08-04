from tbparse import SummaryReader
import pandas as pd

import numpy as np

def load_dataframe_tensors():

    log_dir = "../test_logs"
    reader = SummaryReader(log_dir)

    df = reader.tensors

    # Rename
    df = df.rename(columns={'step': 'Epoch'})

    df = df.set_index(['Epoch'])

    # For each tag - there must be a column
    tags = df.loc[:, "tag"].unique()

    data = {}

    for tag in tags:
        mask = df["tag"] == tag
        
        df_tmp = df.loc[mask]
        
        new_tag = tag.replace("_", " ")

        data[new_tag] = df_tmp.value 

    df = pd.DataFrame(data)
    return df


def load_dataframe_text():

    log_dir = "../test_logs"
    reader = SummaryReader(log_dir)

    df = reader.text
   
    # Rename
    df = df.rename(columns={'step': 'Epoch'})

    df = df.set_index(['Epoch'])

    # For each tag - there must be a column
    tags = df.loc[:, "tag"].unique()


    dfs = []
    for tag in tags:
        mask = df["tag"] == tag
        
        df_tmp = df.loc[mask]
        
        new_tag = tag.replace("_", " ")


        df_tag = df_tmp.value

        num_epoch = len(df_tmp)

        for epoch in range(num_epoch):

            x = df_tag[epoch]
            x = x.replace("[", "")
            x = x.replace("]", "")
            x = x.replace("\n", "")
            x = x.split(" ")

            x = [float(num) for num in x if len(num) != 0]
            x = np.array(x)


            for value in x:

                data = {
                    "tag": new_tag, 
                    "Epoch": epoch,
                    "value": value
                }

                df2 = pd.DataFrame(data, index=[0])
        
                dfs.append(df2)
    

    df = pd.concat(dfs)

    return df

 