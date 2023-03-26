# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:32:23 2022

@author: nickj
"""

import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
import time
#from selenium import JavaScriptExecutor
#from selenium import WebElement

class preprocess_users():
    """
    This class exists to preprocess the data extracted from myanimelist.net, so that it's ready to be used
    in a machine learning model, and add additional attributes to the anime data from myanimelist.net which
    weren't added during the initial web crawl.
    """
    def __init__(self):
        self.total_data = pd.DataFrame(columns=['Name','Score','UserID'])
        self.numusers = 0
        self.path1 = 'users/'
        self.path2 = 'users2/'
        
    def create_dataframe(self):
        self.load_in_data(self.path1)
        self.load_in_data(self.path2)
        self.total_data.to_csv('total_data.csv')
        
    def load_in_data(self,path):
        for file in os.listdir(path):
            user = pd.read_csv(path + file)
            user = user.loc[:,['Name','Score']]
            user = user[user['Score'] != '-']
            user['UserID'] = file.split('_')[1].split('.')[0]
            user = user[['UserID','Name','Score']]
            self.total_data = self.total_data.append(user,ignore_index=True)
            print(self.total_data.shape)
            
    def filter_anime(self):
        anime_1 = pd.read_csv('animes3.csv',index_col=0)
        anime_2 = pd.read_csv('animes_rest.csv',index_col=0)
        total_anime = anime_1.append(anime_2,ignore_index=True)
        total_anime = total_anime.drop_duplicates(subset='Name',keep='first')
        total_anime = total_anime.reset_index().iloc[:,1:]
        print(total_anime.head())
        total_anime['AnimeID'] = total_anime.index
        total_anime.to_csv('total_anime.csv')
            
    def replace_anime(self,x):
        x = x.strip().lower()
        val = self.anime['AnimeID'][self.anime['Name'].str.lower() == x]
        val2 = self.anime['AnimeID'][self.anime['english'].str.lower() == x]
        self.count += 1
        if self.count%1000 == 0:
            print(x)
            print(val)
            print(val2)
            print(self.count)
        if val.empty and val2.empty:
            return 'Missing'
        else:
            if not val.empty:
                return val.iloc[0]
            if not val2.empty:
                return val2.iloc[0]
    
    def add_missing_anime(self):
        self.missing = pd.read_csv('animes_missed.csv',index_col=0)
        self.anime = pd.read_csv('anime_eng_titles.csv',index_col=0)
        self.missing['english'] = ''
        for i in self.missing.index:
            print(i)
            url = self.missing.loc[i,'url']
            html = urlopen(url)
            bsObj = BeautifulSoup(html,'html.parser')
            title_eng = bsObj.find('p',class_='title-english')
            if title_eng == None:
                continue
            else:
                title_eng = title_eng.get_text()
                self.missing.loc[i,'english'] = title_eng.strip()
                print(title_eng.strip())
                #self.missing.to_csv('anime_eng_titles.csv')
                time.sleep(1)
        self.anime = self.anime.append(self.missing,ignore_index=True)
        self.anime = self.anime.reset_index().iloc[:,1:]
        self.anime.to_csv('all_anime_final.csv')
        
            
    def remove_music(self):
        anime = pd.read_csv('all_anime_final.csv',index_col=0)
        anime['AnimeID'] = anime.index
        print(anime.shape)
        print(anime.Type)
        anime = anime[anime.Type != "Music"]
        print(anime.shape)
        #print(anime.head())
        anime.to_csv('all_anime_final.csv')
    
    
    def filter_users(self):
        self.count = 0
        users = pd.read_csv('total_data.csv',index_col=0)
        self.anime = pd.read_csv('all_anime_final.csv',index_col=0)
        users['AnimeID'] = users.apply(lambda x: self.replace_anime(x[0]),axis=1)
        users.to_csv('users_clean.csv')
        
    def remove_missing(self):
        users = pd.read_csv('users_clean.csv',index_col=0)
        users = users[~[users.loc['AnimeID'] == 'Missing']]
        print(users.shape)
        users.to_csv('users_clean.csv')
        
    def get_english_names(self):
        #anime = pd.read_csv('total_anime.csv',index_col=0)
        #anime['english'] = ''
        anime = pd.read_csv('anime_eng_titles.csv',index_col=0)
        for i in anime.index:
            #if i > 25:
               # continue
            print(i)
            url = anime.loc[i,'url']
            html = urlopen(url)
            bsObj = BeautifulSoup(html,'html.parser')
            title_eng = bsObj.find('p',class_='title-english')
            if title_eng == None:
                continue
            else:
                title_eng = title_eng.get_text()
                anime.loc[i,'english'] = title_eng.strip()
                print(title_eng.strip())
                anime.to_csv('anime_eng_titles.csv')
                time.sleep(1)
           
    def more_anime_attributes(self):
        anime = pd.read_csv('anime_modified.csv',index_col=0)
        
        for i,row in anime['url'].iteritems():
            if i < 4853:
                continue
            print(f"Step {i} of {anime.shape[0]}")
            print(row)
            html = urlopen(row)
            bsObj = BeautifulSoup(html,'html.parser')
            
            themespan = bsObj.find("span",text="Themes:")
            if themespan == None:
                themespan = bsObj.find("span",text="Theme:")
            if themespan == None:
                themes = 'None'
            else:
                themeparent = themespan.parent
                themelist = themeparent.findAll("a")
                themes = ""
                for val in themelist:
                    themes = themes + val.getText() + '/'
                themes = themes[0:len(themes)-1]
                
            anime.loc[i,'Themes'] = themes
            
            actors = bsObj.find_all('td',class_='va-t')
            if actors == None:
                continue
            
            totalactors = ""
            for actor in actors:
                totalactors = totalactors + actor.find('a').get('href') + '\n'
            anime.loc[i,'Actors'] = totalactors
            anime.to_csv('anime_modified.csv')
    
    def parent_story(self):
        anime = pd.read_csv('anime_FINAL.csv',index_col=0)
        #anime['Parent Story'] = 'None'
        for i,row in anime['url'].iteritems():
            if i < 2846:
                continue
            print(f"Step {i} of {anime.shape[0]}")
            print(row)
            html = urlopen(row)
            bsObj = BeautifulSoup(html,'html.parser')
            pstory_td = bsObj.find("td",text="Parent Story:")
            if pstory_td == None:
                pstory = 'None'
            else:
                pstory_parent = pstory_td.parent
                pstory = pstory_parent.find("a").get("href")
                anime.loc[i,'Parent Story'] = pstory
            anime.to_csv('anime_FINAL.csv')
        

def main():
    users = preprocess_users()
    #users.create_dataframe()
    #users.filter_anime()
    #users.filter_users()
    #users.get_english_names()
    #users.add_missing_anime()
    #users.remove_music()
    #users.more_anime_attributes()
    users.parent_story()
    
main()
        