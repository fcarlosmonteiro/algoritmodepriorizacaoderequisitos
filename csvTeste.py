# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:14:47 2018

@author: jeanc
"""

import pandas as pd

bd = pd.read_csv('base.csv')

#for row in base_dados.iterrows():
    #print(row)
    
bd1 = bd.iloc[0:5,0:4]
bd2 = bd.iloc[5:10,0:4]
bd3 = bd.iloc[10:15,0:4]
bd4 = bd.iloc[15:20,0:4]

