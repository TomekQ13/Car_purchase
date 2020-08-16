#importing all necessary packages
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

from statsmodels.stats.multicomp import pairwise_tukeyhsd

import re
import nltk


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from datetime import datetime

import xgboost as xgb
from xgboost import XGBRegressor

#custom functions
def clear():
    os.system('cls')

def ListNoDups(mylist):
    mylist = list(dict.fromkeys(mylist))
    return mylist

def CleanList(list_to_clean, len_less_than = 3):
    cleaned_list = [x for x in list_to_clean if str(x) != 'nan' and len(x) >= len_less_than ]
    return cleaned_list

def CombineTwoLists(list1, list2):
    #Combines two lists i.e. element 0 of list 1 to elemenet 0 of list 2 and creates a list so the output is a list of lists
    
    if type(list2) != list or type(list1) != list:
        print('At least one argument is not a list')
        return        
    
    if len(list1) != len(list2):
        print('Lists have different lengths')
        return
    
    final_list = []
    
    for index, (element1, element2)  in enumerate(zip(list1, list2)):
        list_to_append = [element1, element2]
        final_list.append(list_to_append)
    
    final_list = final_list[0:]
    return final_list

def SplitList(list):
    final_list = []
    for _unused_,element2 in enumerate(list):
        final_list.append(element2[0])
    
    return final_list

#class
class Car:
    
    missing = -1
    duplicates = -1
    
    price_outliers = -1
    mileage_outliers = -1
    year_outliers = -1
    total_discard = -1
    
    corpus = []
    def __init__(self, path = '', price_outlier_mt = 200000, mileage_outlier_mt = 400000,
                year_outlier_lt = 1995, engine_outlier_mt = 4000, engine_outlier_lt = 750,
                dependent_variable = 'price',
                categorical_variables = ['engine_type', 'city', 'province'],
                numeric_variables = ['price', 'mileage_km', 'engine_cm3', 'year'],
                read_from_path = 'yes', pandas_dataframe = '_NULL_'):
        
        #define outliers values
        self.price_outlier_mt = price_outlier_mt
        self.mileage_outlier_mt = mileage_outlier_mt
        self.year_outlier_lt =  year_outlier_lt
        self.engine_outlier_mt = engine_outlier_mt
        self.engine_outlier_lt = engine_outlier_lt
        self.dependent_variable = dependent_variable       

        
        
        #define variable data types
        self.numeric_variables = numeric_variables
        self.categorical_variables = categorical_variables
        
        #read all .csv files from the directory
        self.read_from_path = read_from_path
        self.pandas_dataframe = pandas_dataframe
        
        #administrative variables
        self.outliers_removed = 0
        
        if self.read_from_path == 'yes':        
            self.data = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv"))), sort=False)
        else:
            self.data = self.pandas_dataframe
        
        #drop the duplicates and save the number of duplicates - many duplicates due to data gathering method
        self.duplicates = len(self.data) - len(self.data.drop_duplicates())
        self.data.drop_duplicates(inplace = True)
        
        #rename columns
        self.data.columns = ['title', 'price', 'sub_title', 'mileage_km', 'year', 'engine_cm3',
                'engine_type', 'city', 'province', 'negotiable']
        
        #drop NaNs and save the number of rows dropped to the missing varaible
        self.missing = self.data['engine_type'].isna().sum()
        self.data.dropna(subset = ['engine_type'], axis = 'index', inplace = True)
        
        self.missing = self.missing + self.data['city'].isna().sum()
        self.data.dropna(subset = ['city'], axis = 'index', inplace = True)
        
        self.missing = self.missing + self.data['engine_cm3'].isna().sum()
        self.data.dropna(subset = ['engine_cm3'], axis = 'index', inplace = True)
        
        #clean up the columns
        self.data['price'] = self.data['price'].apply(lambda x: x.replace(",", ""))
        self.data['price'] = self.data['price'].apply(lambda x: x.replace(" ", "")).astype('int')
        
        self.data['mileage_km'] = self.data['mileage_km'].apply(lambda x: x.replace("km", ""))
        self.data['mileage_km'] = self.data['mileage_km'].apply(lambda x: x.replace(" ", "")).astype('float')
        
        self.data['engine_cm3'] = self.data['engine_cm3'].astype('str')
        self.data['engine_cm3'] = self.data['engine_cm3'].apply(lambda x: x.replace('cm3', ''))
        self.data['engine_cm3'] = self.data['engine_cm3'].apply(lambda x: x.replace(' ','')).astype('int')
        
        self.data['province'] = self.data['province'].astype('str')
        self.data['province'] = self.data['province'].apply(lambda x: x.replace('(',''))
        self.data['province'] = self.data['province'].apply(lambda x: x.replace(')',''))
        
        self.data['sub_title'] = self.data['sub_title'].astype('str') #may change that in the future - possible info loss due to lowercase
        
        
        self.data['title'] = self.data['title'].astype('str') #may change that in the future - possible info loss due to lowercase
        
        self.data['negotiable'] = self.data['negotiable'].astype('str')
        
        #Add ID column
        self.data.insert(loc = 0, column = 'ID', value = range(1, len(self.data)+1))

        '''
        #discard outliers and calculate the numbers
        self.total_discard = len(self.data) - len(self.data[(self.data['price'] <= self.price_outlier_mt) &
                                                        (self.data['mileage_km'] <= self.mileage_outlier_mt) &
                                                        (self.data['year'] >= self.year_outlier_lt) &
                                                        (self.data['engine_cm3'] <= self.engine_outlier_mt) &
                                                        (self.data['engine_cm3'] >= self.engine_outlier_lt)])        
        
        self.price_outliers = len(self.data[self.data['price'] > price_outlier_mt])
        self.data = self.data[self.data['price'] <= price_outlier_mt]
        
        self.mileage_outliers = len(self.data[self.data['mileage_km'] > mileage_outlier_mt])
        self.data = self.data[self.data['mileage_km'] <= mileage_outlier_mt]
        
        self.year_outliers = len(self.data[self.data['year'] < year_outlier_lt])
        self.data = self.data[self.data['year'] >= year_outlier_lt]
        
        self.engine_outliers = len(self.data[(self.data['engine_cm3'] > engine_outlier_mt) |
                                            (self.data['engine_cm3'] < engine_outlier_lt)])
        self.data = self.data[(self.data['engine_cm3'] <= engine_outlier_mt) & 
                             (self.data['engine_cm3'] >= engine_outlier_lt)]
        '''
        
        #NLP
        self.data['concat_title_subtitle'] = self.data['title'] + ' ' + self.data['sub_title']
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.lower())
        
        #replace problematic cases for NLP for title and subtitle
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('+',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('(',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace(')',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('**',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('*',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace(']',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('[',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace("/"," "))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace("\\"," "))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace(',',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('?',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('.',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('!',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('_',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('-',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('|',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('#',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('%',' '))
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('~',' '))  
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('*',' ')) 
        self.data['concat_title_subtitle'] = self.data['concat_title_subtitle'].apply(lambda x: x.replace('*',' '))
        
        #NLP preprocessing for location
        self.data['province'] = self.data['province'].apply(lambda x: x.lower()) 
        self.data['province'] = self.data['province'].apply(lambda x: x.replace('ą', 'a')) 
        self.data['province'] = self.data['province'].apply(lambda x: x.replace('ę', 'e'))
        self.data['province'] = self.data['province'].apply(lambda x: x.replace('ł', 'l'))
        self.data['province'] = self.data['province'].apply(lambda x: x.replace('ś', 's'))
        self.data['province'] = self.data['province'].apply(lambda x: x.replace('ć', 'c'))
        self.data['province'] = self.data['province'].apply(lambda x: x.replace('ż', 'z'))
        
    def describe(self):
        #descriptive statistice
        desc_stats = round(pd.DataFrame(
                        data = self.data[self.numerical_variables].describe(),
                        columns = self.data[self.numerical_variables].columns),2)
        return desc_stats
    
    def remove_outliers(self, variable, value, mode):
        if variable not in self.numeric_variables:
            print('Specify a numerical variable')
            return
        
        if mode not in ['more_than', 'less_than']:
            print('Wrong mode. Specify more_than or less_than')
            return
        else:
            self.original_length = len(self.data)
        
        if mode == 'more_than':
            self.data = self.data[self.data[variable] <= value]
            return self        
        elif mode == 'less_than':
            self.data = self.data[self.data[variable] >= value]
            return self
        
        #set a value to the administrative variable
        self.outliers_removed = 1
            
        self.total_discard = len(self.data) - self.original_length
    
    def outliers(self):
        #baisc data about discarded outliers
        if self.outliers_removed == 0:
            print('Remove outlier first using remove_outliers method')     
        elif self.outliers_removed == 1:
            print('Offers with price greater than '+str(self.price_outlier_mt)+' have been discarded')
            print('The number of such offers = '+str(self.price_outliers))
            print('')
            print('Offers with mileage greater than '+str(self.mileage_outlier_mt)+' have been discarded')
            print('The number of such offers = '+str(self.mileage_outliers))
            print('')
            print('Offers with year lower than '+str(self.year_outlier_lt)+' have been discarded')
            print('The number of such offers = '+str(self.year_outliers))
            print('')
            print('Offers with engine_cm3 greater than '+str(self.engine_outlier_lt)+
                  ' and lower than '+str(self.engine_outlier_mt)+' have been discarded')
            print('The number of such offers = '+str(self.engine_outliers))
            print('')
            print('Total number of discarded offers = '+str(self.total_discard)
                  +'('+str(round(self.total_discard/len(self.data)*100,2))+'%)'
                  +' - may be different to the sum of above due to overlap')
        
    def scatter_nox(self, var = 'all', figsize_1 = 7, figsize_2 = 5):
        #prints scatter plots with no x axis - a dummy sequence as x axis
        if var != 'all' and var not in self.data.columns:
            print('Variable not found in the dataset')
        if var == 'all':
            plt.rcParams["figure.figsize"] = (figsize_1,figsize_2)
            plt.scatter(y = self.data['mileage_km'], x = range(1, len(self.data)+1), s=1)
            plt.title('mileage_km')
            plt.show()

            plt.scatter(y = self.data['price'], x = range(1, len(self.data)+1), s=1)
            plt.title('price')
            plt.show()

            plt.scatter(y = self.data['year'], x = range(1, len(self.data)+1), s=1)
            plt.title('year')
            plt.show()

            plt.scatter(y = self.data['engine_cm3'], x = range(1, len(self.data)+1), s=1)
            plt.title('engine_cm3')
            plt.show()
        else:
            plt.rcParams["figure.figsize"] = (figsize_1,figsize_2)
            plt.scatter(y = self.data[var], x = range(1, len(self.data)+1), s=1)
            plt.title(var)
            plt.show()
            
    def scatter(self, var = 'all'):
        #prints scatter plots for numerical variables
        if var != 'all' and var not in self.data.columns:
            print('Variable not found in the dataset')
        list_comb = []
        if var == 'all':
            for variable1 in enumerate(self.numeric_variables):
                for variable2 in enumerate(self.numeric_variables):
                    if variable1 != variable2 and variable1[1]+variable2[1] not in list_comb and variable2[1]+variable1[1] not in list_comb:
                        plt.scatter(y = self.data[variable1[1]], x = self.data[variable2[1]], s=1)
                        plt.title("Correlation between "+variable1[1]+' and '+variable2[1])
                        plt.ylabel(variable1[1])
                        plt.xlabel(variable2[1])
                        plt.show()
                        list_comb.append(variable1[1]+variable2[1])
        #else: - TO DO
    
    def hist(self, var = 'all', bins = 50):
        if var != 'all' and var not in self.data.columns:
            print('Variable not found in the dataset')
        #prints histograms for numeric variables
        if var == 'all':
            for variable in enumerate(self.numeric_variables):
                plt.hist(x = self.data[variable[1]], bins = bins)
                plt.title(variable[1])
                plt.show()
     
    def price_cat_vars(self, variables = '_NULL_'):
        
        if variables == '_NULL_':
            variables = self.categorical_variables
            
        for variable in enumerate(variables):
            # shows desrptive statistics of categorical variables
            print(self.data.groupby(self.data[variable[1]])['price'].describe())
            #the variables need further preprocessing
            
    def add_dummies(self, columns_to_check, categorical_list = '_NULL_', delete_from_strings = 'yes', delete_column = 'no'):
        #adds dummmies from cat_list, checks in every column of columns_to_check

        for column in enumerate(columns_to_check):            
            if categorical_list == '_NULL_':
                categorical_list = self.data[column[1]].unique().tolist()
            
            for category in enumerate(categorical_list):
                col_name = column[1] + '_' + category[1]
                self.data[col_name] = self.data[column[1]].str.contains(category[1]).astype('int')
                
                #append newly craeted variables to categorical variables
                if self.data[col_name].sum() > 0:
                    self.categorical_variables.append(col_name)
                else:
                    self.data.drop(columns = [col_name], inplace = True)
                
                #delete the string from the column
                if delete_from_strings == 'yes':
                    self.data[column[1]] = self.data[column[1]].apply(lambda x: x.replace(category[1], ''))
                    
            #delete the column
            if delete_column == 'yes':
                self.data.drop(columns = column[1], inplace = True)
                
        return self
    
    def add_dummies2(self, categorical_list, delete_from_column = 'yes', delete_concat_column = 'no', column = 'concat_title_subtitle'):
        #adds dummmies from cat_list, checks in concat_title_subtitle column
        for category in enumerate(categorical_list):
            
            #print(category)
            
            col_name = category[1]
            self.data[col_name] = self.data[column].str.contains(category[1]).astype('int')
                
                #append newly craeted varaibles to categorical variables
            if self.data[col_name].sum() > 0:
                self.categorical_variables.append(col_name)
            else:
                self.data.drop(columns = [col_name], inplace = True)
                
                #delete the string from the column
            if delete_from_column == 'yes':
                self.data[column] = self.data[column].apply(lambda x: x.replace(category[1], ''))
                                                
            if delete_concat_column == 'yes':
                self.data.drop(columns = column, inplace = True)
                
        return self
    """
    def ind_test(self, var, alpha = 0.05):
        if alpha > 1 or alpha < 0:
            print('Incorrect alpha value. Select a value from <0;1>.')
            
        if var != 'all' and var not in self.data.columns:
            print('Variable not found in the dataset')
        pivot = round(self.data.pivot_table(values = 'price', index = var, aggfunc = ['count', 'mean']),2)
        pivot.columns = ['count', 'mean']
        
        mean_price = self.data['price'].mean()
        
        pivot['sm'] = pivot['mean']/((pivot['count'])**(1/2))
        
        pivot['t'] = (pivot['mean']-mean_price)/pivot['sm']
        pivot['df'] = pivot['count']-1

        #calculate p-value
        pivot['t_border'] = stats.t.ppf(1-alpha/2, pivot['df'])
        
        #implementation here is not 100% mathematically correct
        return pivot
    """
    def anova(self, var = 'all', alpha = 0.05):
        if var == 'all':
            for variable in enumerate(self.categorical_variables):
                anova_data = self.data[[variable[1], 'price']].reset_index().copy()
                anova_data.columns = ['index', variable[1], 'price']
                equation_string = 'price ~ '+str(variable[1])
                model = ols(equation_string, data=anova_data).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                print(anova_table)
                print()
                
                #pairwise comparisons
                pairwise_comparison = pairwise_tukeyhsd(endog = anova_data['price'],
                                                        groups = anova_data[variable[1]],
                                                        alpha = alpha)
                print(pairwise_comparison)
                print()
        #else:
            #TO DO
            #also TO DO check Anova assumptions
    def make_corpus(self, column = 'concat_title_subtitle'):
        #creates a corpus out of title and subtitle column
        for i in range(0, len(self.data)):
            #string = re.sub('[^a-zA-Z]', ' ', self.data.reset_index().loc[i]['concat_title_subtitle'])
            string = self.data.reset_index().loc[i][column]
            string = string.split()
            self.corpus = self.corpus + string
        self.corpus = ListNoDups(self.corpus)
        
        #with open("corpus.txt", "w") as output:
         #   output.write(str(self.corpus))
    
        return self.corpus
    
    def analyse_variables(self, list_of_variables, discard = 0.01):
    # independence tests for a list of variable e.g. corpus
        final_df = pd.DataFrame(columns = ['variable', 'mean_1', 'mean_0', 'count_1', 'count_0'])

        for variable in enumerate(list_of_variables):
            #debug
            #print(str(variable)+' done')
            
            self.data[variable[1]] = self.data['concat_title_subtitle'].str.contains(variable[1]).astype('int')

            mean_1 = self.data.loc[self.data[variable[1]] == 1][self.dependent_variable].mean()
            mean_0 = self.data.loc[self.data[variable[1]] == 0][self.dependent_variable].mean()

            count_1 = len(self.data.loc[self.data[variable[1]] == 1])
            count_0 = len(self.data.loc[self.data[variable[1]] == 0])        
            
            if count_1 >= discard * len(self.data) and count_0 >= discard * len(self.data):            
                dict_to_append = {
                    'variable' : variable[1],
                    'mean_1' : mean_1,
                    'mean_0' : mean_0,
                    'count_1' : count_1,
                    'count_0' : count_0
                }

                final_df = final_df.append(dict_to_append, ignore_index = True)

            self.data.drop(columns = [variable[1]], inplace = True)

            #if variable[0] % 1000 == 0:
            #    print(str(variable[0])+'/'+str(len(list_of_variables)))         
            
        
        
        final_df['mean_diff'] = abs(final_df['mean_1'] - final_df['mean_0'])
        final_df = final_df.sort_values(by = 'mean_diff', ascending = False).reset_index()
        final_df.drop(columns = 'index', inplace = True)
        
        date = datetime.date(datetime.now())
        
        final_df.to_csv('analyse_variables_'+str(date)+'.csv')
        
        return final_df

        
            

#cv = CountVectorizer(max_features = 1000)
#X = cv.fit_transform(corpus).toarray()