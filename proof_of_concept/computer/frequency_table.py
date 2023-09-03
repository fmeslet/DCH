#!/home/fmeslet/anaconda3/bin/python3.8
#-*-coding: utf-8 -*-


###############
# IMPORTATIONS
###############


import numpy as np
import pandas as pd


###############
# CODE
###############


class FrequencyTable():
    def __init__(self, 
                 df_table, 
                 index_pos,
                 alphabet_size,
                 cut_value):
        self.df_table = df_table
        self.index_pos = index_pos
        self.alphabet_size = alphabet_size
        self.cut_value = cut_value
        
    def predict(self, 
                data, 
                pos,
                ctx,
                soft=True):
        
        pred_proba = np.ones(
            (data.shape[0], self.alphabet_size))*0.5
        
        for i in range(data.shape[0]):

            print("\n[DEBUG][predict] i: ", i)
            print("[DEBUG][predict] ctx[i], pos[i]: ", ctx[i], pos[i])
            print("[DEBUG][predict] self.index_pos[ctx[i], pos[i]]: ", 
                    self.index_pos[ctx[i], pos[i]])
            print("[DEBUG][predict] data.shape: ", data.shape)
        
            # Extract indexes
            indexes_values_extract = \
                self.index_pos[
                    ctx[i], pos[i]]

            print("[DEBUG][predict] values_extract: ", indexes_values_extract)

            #print("[DEBUG][predict] self.index_pos: ", 
            #            self.index_pos)

            print("[DEBUG][predict] ctx[i], pos[i]: ", ctx[i], pos[i])
            print("[DEBUG][predict] self.index_pos[ctx[i], pos[i]] : ", self.index_pos[
                    ctx[i], pos[i]])

            # Remove -1 when mode is OPTIMAL or SELECTIVE
            indexes_values_extract = \
                indexes_values_extract[
                    indexes_values_extract >= 0]

            print("[DEBUG][predict] indexes_values_extract: ", indexes_values_extract)
            print("[DEBUG][predict] data.shape: ", data.shape)

            # Extract values
            values_extract = data[
                    i, indexes_values_extract]

            print("[DEBUG][predict] values_extract BEFORE PADDING: ", values_extract)
            print("[DEBUG][predict] self.cut_value BEFORE PADDING: ", self.cut_value)

            if (values_extract.size < self.cut_value):

                # Set 0 at the beginning
                values_extract = np.lib.pad(values_extract,
                        (0, self.cut_value-values_extract.size),
                        'constant', constant_values=(0))
                
                print("[DEBUG][predict] values_extract AFTER PADDING: ", values_extract)

                

            # Convert bit to str
            values_extract = values_extract.ravel().astype(
                    np.uint8).astype(str)

            print("[DEBUG][predict] values_extract : ", values_extract)

            print("[DEBUG][decompression_packet] data[0, -20:] : ", data[0, -20:].astype(np.uint8))

            # Extract bit 
            values_extract = "".join(
                values_extract)

            #print("[DEBUG][predict] values_extract : ", values_extract)
            
            values_extract = str(values_extract)

            #print("[DEBUG][predict] values_extract : ", values_extract)
            
            # Get values
            index_pos = self.df_table.loc[
                (ctx[i],)].index.get_level_values(0)

            #print("[DEBUG][TableModel][predict] index_pos : ", index_pos)
            #print("[DEBUG][TableModel][predict] np.isin(pos[i], index_pos).any() : ", 
            #       np.isin(pos[i], index_pos).any())

            if (np.isin(pos[i], index_pos).any()):

                #print("[DEBUG][predict] INSIDE IF : ")

                #print("[DEBUG][TableModel][predict][if] self.df_table.loc[(ctx[i], )] : ", 
                #       self.df_table.loc[(ctx[i], )])

                #print("[DEBUG][predict] ctx[i] : ", ctx[i])
                #print("[DEBUG][predict] pos[i] : ", pos[i])

                df_tmp = self.df_table.loc[
                    (ctx[i], #[0], 
                     pos[i], #[0], 
                     )]
            
                # Apply cond
                cond = (values_extract == df_tmp.index.values)
                is_value = (df_tmp[cond].shape[0] == 0)
                
                if (not is_value):
                    # Pour être en accord avec la transformation effectuées
                    # pour la génération des tables
                    proba = int(np.around(
                        df_tmp[cond].values,
                        decimals=2, out=None)*100)
                    proba = proba / 100
                    #proba = np.around(
                    #    df_tmp[cond].values, 
                    #    decimals=2, out=None)
                    pred_proba[i:i+1] = [[proba, 1-proba]]
                    #pred_proba[i:i+1] = [[0.5, 1-0.5]]
                        #[df_tmp[cond].values, 
                        # 1-df_tmp[cond].values]]
                else:
                    if ((not soft)):
                        mean_val = self.df_table.loc[
                            (ctx[i], pos[i], )].mean()
                        pred_proba[i:i+1] = [
                            [mean_val, 
                             1-mean_val]]  

                  
        print("[DEBUG][predict] pred_proba : ", pred_proba)          
                    # Ou utiliser le MAX ?
        return pred_proba
