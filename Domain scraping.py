# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 11:16:19 2021

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

#driver.get('https://www.domain.com.au/sale/melbourne-region-vic/house/?excludeunderoffer=1')

#prop_list_page=range(5,17)

#Get links
#for i in range(0,20):
    # try:
    #     xpath=f'//*[@id="skip-link-content"]/div[1]/div[2]/ul/li[{i}]/div/div[2]/div/a'
    #     raw_link=driver.find_element_by_xpath(xpath)
    #     link=raw_link.get_attribute('href')
    #     link_list.append(link)
    # except:
    #     pass

#Get links from the other pages

for i in range(1,300):
    driver.get(f'https://www.domain.com.au/sale/melbourne-region-vic/house/?excludeunderoffer=1&page={i}')

    #Get links
    for i in range(0,30):
        try:
            xpath=f'//*[@id="skip-link-content"]/div[1]/div[2]/ul/li[{i}]/div/div[2]/div/a'
            raw_link=driver.find_element_by_xpath(xpath)
            link=raw_link.get_attribute('href')
            link_list.append(link)
        except:
            pass

#Get property descriptions

property_descriptions={'url':[],"description":[]}

#Search links for descriptions
for i in range(len(link_list)):
    url=link_list[i]
    driver.get(url)
    property_string=""
    
    #Click on read more button
    button = driver.find_element_by_class_name("css-1pn4141")
    driver.execute_script("arguments[0].click();", button)
    
    #find description
    
    #try xpath with first format
    try:
        xpath='/html/body/div[1]/div/div[1]/div/div[6]/div/div/div[3]/div[1]/div/div[1]/div/p'
        raw_description=driver.find_element_by_xpath(xpath).text
        property_string=property_string+" "+raw_description
    except:
        pass
    
    for i in range(0,20):
        try:   
            #try xpath with second format
            xpath=f'/html/body/div[1]/div/div[1]/div/div[6]/div/div/div[3]/div[1]/div/div/div/p[{i}]'
            raw_description=driver.find_element_by_xpath(xpath).text
            property_string=property_string+" "+raw_description
        
        except:
            pass
    property_descriptions['url'].append(url)
    property_descriptions['description'].append(property_string)

driver.quit()

descriptions_df=pd.DataFrame(property_descriptions)

descriptions_df = descriptions_df[descriptions_df.description != ""]

descriptions_df.to_csv("descriptions.csv",index=False)
