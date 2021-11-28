# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:04:46 2021

@author: Darcy Richardson
"""
#Importing packages
from selenium import webdriver
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome('C:\\Users\\Darcy Richardson\\.wdm\\drivers\\chromedriver\\win32\\96.0.4664.45\\chromedriver.exe')


link_list=[]

#Get links from first listing page

base_url='https://forums.whirlpool.net.au/thread/3qqz5jp3'


#Get page length

        

#Get comments from forums 
comments_list={'comment':[]}

#Scroll through pages of comments
for x in range(1,661):    
    try:
        url=base_url+f"?p={x}"
        
        driver.get(url)
            
        #Scroll through comments on page
        for y in range(1,30):
            try:
                xpath=f'/html/body/div/div[2]/div[3]/div[2]/div[2]/div[3]/div[{y}]/div/div[2]/p'
                comment_elements = driver.find_elements_by_xpath(xpath)
                comments = [comment.text for comment in comment_elements]
                comments=''.join(comments)
                comments_list['comment'].append(comments)
            except:
                pass
    except:
        break

driver.quit()

comments_df=pd.DataFrame(comments_list)

comments_df = comments_df[comments_df.comment != ""]

comments_df.to_csv("comments.csv",index=False)


