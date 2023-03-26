# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:41:55 2022

@author: nickj
"""

from bs4 import BeautifulSoup
import numpy as np
#import sklearn
import pandas as pd
from urllib.request import urlopen
import os
from unidecode import unidecode
import time
import matplotlib.pyplot as plt
from selenium import webdriver

class Collect_Anime:
    """
    This is the main class for collecting the data on anime series.
    I use the urllib and BeautifulSoup libraries to scrape and extract information from
    the pages of myanimelist.net, storing it in a Dataframe and saving it to an Excel file.
    """
    def __init__(self):
        self.pageurl = 'https://myanimelist.net/topanime.php'
        #self.frame = pd.DataFrame(columns=['url','Name','Rating','Popularity','Studio','Type','Date','Genres','Adaptation','Prequel'])
        self.frame = pd.read_csv('animes_missed.csv',index_col=0)
        self.compareto = pd.read_csv('anime_eng_titles.csv',index_col=0)
        self.number = 0
        self.pages = 0
        self.failedvals = []
        
    def isAscii(self,s):
        """
        Determines whether a string contains all ASCII characters.
        """
        return all(ord(c) < 128 for c in s)
    
    def getData(self):
        """
        This function loops through the pages of the myanimelist top anime list.
        It uses BeautifulSoup to identify the relevant HTML tags to reach the page
        of a specific anime, then calls the get_anime function to extract the data.

        """
        while(1):
            if self.pages > 0:
                url = self.pageurl + '?limit=' + str(50*self.pages)
            else:
                url = self.pageurl
            self.pages += 1
            html = urlopen(url)
            bsObj = BeautifulSoup(html,'html.parser')
            #for i in range(20):
            threadlist = bsObj.findAll("h3", class_="fs14") #selects anime pages
            #print(threadlist)
            if len(threadlist) == 0:
                break
            for thread in threadlist:
                self.number += 1
                print(self.number)
                if self.number < 0:
                    continue
                vals = self.get_anime(thread)
                if vals == 0:
                    continue
                self.frame = self.frame.append(vals,ignore_index=True)
                #print(vals)
                
                self.frame.to_csv('animes_missed.csv')
        
    def get_anime(self,element):
        """
        Gets the data for a specific anime series using BeautifulSoup.

        Parameters
        ----------
        element : HTML element for an anime page on myanimelist.net

        Returns
        -------
        A dictionary object containing the relevant information relating to an anime series.

        """
        page = element.find("a").attrs['href']
        page = unidecode(page)
        print(page)
        #print(self.compareto['url'].iloc[0])
        if page == None:
            print("Error")
        elif any(self.compareto['url'].eq(page)):
            print("Not New")
            return
        else:
            print(page)
            if not self.isAscii(page):
                page = unidecode(page)
            #print(page)
            html = urlopen(page)
            obj = BeautifulSoup(html,'html.parser')
            name = obj.find("h1",class_="h1_bold_none").getText()
            #print(obj.find_all("span"))
            
            rating = obj.find("div",class_="score-label").getText()
            
            popularity = obj.find("span",class_="members").find("strong").getText()
            
            studiospan = obj.find("span",text="Studios:")
            studioparent = studiospan.parent
            studio = studioparent.find("a").getText()
            
            typespan = obj.find("span",text="Type:")
            typeparent = typespan.parent
            typeval = typeparent.getText().split()[1]
            
            datespan = obj.find("span",text="Aired:")
            dateparent = datespan.parent
            dateval = dateparent.getText().strip()
            datelist = dateval.split()
            if len(datelist) > 3:
                date = datelist[3]
            else:
                date = datelist[len(datelist)-1]
            #print(date)
            
            genrespan = obj.find("span",text="Genres:")
            if genrespan == None:
                genrespan = obj.find("span",text="Genre:")
            if genrespan == None:
                genres = 'None'
            else:
                genreparent = genrespan.parent
                genrelist = genreparent.findAll("a")
                genres = ""
                for val in genrelist:
                    genres = genres + val.getText() + '/'
                genres = genres[0:len(genres)-1]
            
            adaptation_td = obj.find("td",text="Adaptation:")
            if adaptation_td == None:
                adaptation = 'None'
            else:
                adaptation_parent = adaptation_td.parent
                adaptation = adaptation_parent.find("a").get("href")
                
            prequel_td = obj.find("td",text="Prequel:")
            if prequel_td == None:
                prequel = 'None'
            else:
                prequel_parent = prequel_td.parent
                prequel = prequel_parent.find("a").get("href")
            
            return ({'url':page,'Name':name,'Rating':rating,'Popularity':popularity,'Studio':studio,
                    'Type':typeval,'Date':date,'Genres':genres,'Adaptation':adaptation,'Prequel':prequel})

class userCollection:
    """
    This class collects random usernames from the myanimelist.net search function. You can collect
    20 users at one time, so the getUserUrls method needs to be called as many times as
    (desired # of users)/20
    """
    def __init__(self):
        self.pageurl = "https://myanimelist.net/users.php"

    def remove_non_ascii(self,s):
        return s.encode('ascii','ignore').decode()

    #call this method 55 times to get 1100 users, and store them in a text file
    def getUserUrls(self):
        """
        Collects usernames from MAL and writes them to a text file.

        """
        html = urlopen(self.pageurl)
        bsObj = BeautifulSoup(html,'html.parser')
        userList = bsObj.findAll("td",class_="borderClass")
        userfile = open('testusers2.txt','a')
        for user in userList:
            val = user.find("a").getText()
            val = val + '\n'
            print(val)
            userfile.write(val)
            #useranimes = "https://myanimelist.net/animelist/" + val + "?status=2"

class userData():
    def __init__(self):
        self.usernames = pd.read_table('testusers3.txt',sep='\n',header=None,squeeze=True)
        #self.usernames = pd.read_csv('new_usernames.csv',index_col=0)
        self.redo_list = []
        
    def uniqueUsers(self):
        self.unique = pd.Series(self.usernames.unique())

    def getUserData(self):
        #self.usernames = self.usernames.iloc[:,0]
        self.uniqueUsers()
        for i,user in self.unique.iteritems():
            i = int(i)
            if i < 10657:
                continue
            print(f"Total members: {len(self.unique)}")
            print(f"Member number: {i}")
            self.getTestUser(user,i)
    
    def redo_Users(self):
        self.uniqueUsers()
        for i,user in self.unique.iteritems():
            i = int(i)
            #print(i)
            try:
                fname = "users/User_" + str(i) + ".csv"
                userdata = pd.read_csv(fname,index_col=0)
                if userdata.shape[0]%300 == 0:
                    print(i)
                    print(user)
                    self.getTestUser(user,i)
            except:
                continue
        
        
    
    #gets all animes in 'all' list for a user
    def getTestUser(self,username,number):
        #driver = webdriver.Chrome()
        driver = webdriver.Chrome("C:\\chromedriver.exe")
        url = "https://myanimelist.net/animelist/" + str(username) + "?status=7"
        driver.get(url)
        html = driver.page_source
        bsObj = BeautifulSoup(html,'html.parser')
        badresult = bsObj.findAll(class_="badresult")
        if len(badresult) > 0:
            driver.close()
            print("Blocked")
            return
        
        diff = len(bsObj.findAll("tr",class_="list-table-data")) + len(bsObj.findAll("a",class_="animetitle"))
        print(diff)
        ct = 0
        while (diff%300 == 0) and ct < 12:
            ct += 1
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") #####
            time.sleep(2)
            html = driver.page_source
            bsObj = BeautifulSoup(html,'html.parser')
            diff = len(bsObj.findAll("tr",class_="list-table-data")) + len(bsObj.findAll("a",class_="animetitle"))
            
        html = driver.page_source
        bsObj = BeautifulSoup(html,'html.parser')
        animes = bsObj.findAll("a",class_="animetitle")
        print(len(animes))
        if len(animes) > 0:
            if len(animes[0].getText()) > 0:
                user = pd.DataFrame(columns=['Name','Score'])
                scores = bsObj.findAll("td",class_=["td1", "td2"],width="45")
                if len(scores) == 0:
                    print ('No Scores')
                    driver.close()
                    return
                for i in range(len(animes)):
                    name = animes[i].get('href')
                    score = scores[i].getText().strip()
                    namescore = {'Name':name,'Score':score}
                    user = user.append(namescore,ignore_index=True)
        else:
            print('lel')
            animes = bsObj.findAll("tr",class_="list-table-data")
            user = pd.DataFrame(columns=['Name','Score'])
        #print(animes)
            print(len(animes))
            for anime in animes:
                td = anime.findAll('td',class_=['clearfix'])
                name= ""
                for val in td:
                    val = val.find('a',class_=['link','sort'])
                    #name = name + val.getText()
                    name = val.get('href')
                    #score = anime.find('td',class_='score').getText().strip()
                    score = anime.find('td',class_='score')
                    if score is None:
                        print ('No Scores')
                        driver.close()
                        return
                    score = score.getText().strip()
                    namescore = {'Name':name,'Score':score}
                    user = user.append(namescore,ignore_index=True)
            print(user)
            driver.close()
        filestring = 'usersfinal/User_' + str(number) + '.csv'
        user.to_csv(filestring)
                #animetext = anime.getText().split()
                #for i in range(len(animetext)):

                
def main():
    anime = Collect_Anime()
    #y = anime.getData()
    #uc = userCollection()
    #for i in range(300):
        #time.sleep(5)
        #uc.getUserUrls()
    ud = userData()
    ud.getUserData()
    #ud.redo_Users()
main()