# Instagram Crawler [![Build Status](https://travis-ci.org/huaying/instagram-crawler.svg?branch=master)](https://travis-ci.org/huaying/instagram-crawler)

Built upon this crawler. 
This crawler could fail due to updates on instagram’s website. 

## Install
1. Make sure you have Chrome browser installed.
2. Download [chromedriver](https://sites.google.com/a/chromium.org/chromedriver/) and put it into bin folder: `./inscrawler/bin/chromedriver`
3. Install Selenium: `pip install -r requirements.txt`

## Crawler
### Usage
```
positional arguments:
  -u USERNAME, --username USERNAME
                        instagram's username
```



### Example
```
python crawler.py -u england
python crawler.py -u kayla_itsines
```
Will crawl the first 100 posts of this user and output a JSON file into the profiles folder for this user.