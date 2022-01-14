"""
暂时实现了三个网站的爬虫，封装在三个类里
因为后面两个网站都有反爬虫的措施，工作量比我想象的大一点
主程序在最下面，可以自行更改保存路径
可以更改主程序中实例化的类来改变网站
"""


import requests
import time
from lxml import etree
import os
import re
import json


class UnsplashSpider(object):
    def __init__(self, keyword, headers, save_path):
        self.headers = headers
        self.save_path = save_path
        self.keyword = "-".join(keyword.split())
        self.url = "https://unsplash.com/s/photos/" + keyword

    def run(self):
        if not os.path.exists(self.save_path + self.keyword + "/"):
            os.mkdir(self.save_path + self.keyword + "/")
        response = requests.get(self.url, headers=self.headers)
        html = etree.HTML(response.content)
        img_elements = html.xpath('//img[@class="YVj9w"]')

        no_name_count = 0

        print("Total result count:" + str(len(img_elements)))
        for idx, img_element in enumerate(img_elements[:30]):
            img_url = img_element.attrib["srcset"].split(r',')[-1].split()[0]
            if "alt" in img_element.attrib.keys():
                img_name = img_element.attrib["alt"]
            else:
                img_name = "Unsplash-" + self.keyword + "-"+ str(no_name_count)
                no_name_count += 1
            with open(self.save_path + self.keyword + "/" + img_name + ".jpg", "wb") as f:
                r_img = requests.get(img_url, headers=self.headers)
                f.write(r_img.content)
                f.close()
            print(str(idx + 1) + "/" + "30")
            time.sleep(0.1)


class BaiduSpider(object):
    def __init__(self, keyword, headers, save_path):
        self.headers = headers
        self.save_path = save_path
        self.keyword = "+".join(keyword.split())
        self.kv = {"tn": "baiduimage", "ie": "utf-8", "word": keyword, "pn": "1"}
        self.url = "https://image.baidu.com/search/flip"

    def run(self):
        if not os.path.exists(self.save_path + self.keyword + "/"):
            os.mkdir(self.save_path + self.keyword + "/")
        response = requests.get(self.url, headers=self.headers, params=self.kv)
        img_urls = re.findall('"objURL":"(.*?)",', response.content.decode())

        count = 0

        print(len(img_urls))
        print(type(img_urls))
        print("Total result count:" + str(len(img_urls)))
        for idx, img_url in enumerate(img_urls[:30]):
            img_name = "Baidu-" + self.keyword + "-" + str(count)
            count += 1
            with open(self.save_path + self.keyword + "/" + img_name + ".jpg", "wb") as f:
                r_img = requests.get(img_url, headers=self.headers)
                f.write(r_img.content)
                f.close()
            print(str(idx + 1) + "/" + "30")
            time.sleep(0.1)


class QuanjingSpider(object):
    def __init__(self, keyword, headers, save_path):
        self.headers = headers
        self.headers['Referer'] = 'https://www.quanjing.com/search.aspx?q=%E5%9C%A3%E8%AF%9E'
        self.save_path = save_path
        self.keyword = keyword
        self.kv = {'t': '6570', 'callback': 'searchresult', 'q': keyword, 'stype': '1', 'pagesize': '100', 'pagenum': '1', 'imageType': '2', 'imageColor': '', 'brand': '', 'imageSType': '', 'fr': '1', 'sortFlag': '1', 'imageUType': '', 'btype': '', 'authid': '', '_': '1640268423440'}
        self.url = "https://www.quanjing.com/Handler/SearchUrl.ashx?"

    def run(self):
        if not os.path.exists(self.save_path + self.keyword + "/"):
            os.mkdir(self.save_path + self.keyword + "/")
        response = requests.get(self.url, headers=self.headers, params=self.kv)
        # img_urls = re.findall('"objURL":"(.*?)",', response.content.decode())
        dict_text = json.loads(response.text[13:-1])
        imglist = dict_text['imglist']
        # name = imglist[0]["caption"]

        count = 0


        requests.packages.urllib3.disable_warnings()
        print("Total result count:" + str(len(imglist)))
        for idx, img_dict in enumerate(imglist[:30]):
            img_name = "Quanjing-" + self.keyword + "-" + str(count)
            img_url = img_dict["imgurl"]
            count += 1
            with open(self.save_path + self.keyword + "/" + img_name + ".jpg", "wb") as f:
                r_img = requests.get(img_url, headers=self.headers, verify=False)
                f.write(r_img.content)
                f.close()
            print(str(idx + 1) + "/" + "30")
            time.sleep(0.1)


if __name__ == "__main__":
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/60.0.3112.113 Safari/537.36 '
    }
    save_path = "./spider/"
    if not os.path.exists(save_path[:-1]):
        os.mkdir(save_path[:-1])
    keyword = input("key word:")
    # 选择爬取的网站 UnsplashSpider/BaiduSpider/QuanjingSpider
    spider = UnsplashSpider(keyword=keyword, headers=headers, save_path=save_path)
    spider.run()


# url = "https://www.quanjing.com/Handler/SearchUrl.ashx?" #t=1196&callback=searchresult&q=%E5%9C%A3%E8%AF%9E&stype=1&pagesize=100&pagenum=1&imageType=2&imageColor=&brand=&imageSType=&fr=1&sortFlag=1&imageUType=&btype=&authid=&_=1640262396678"
# kv = {'t': '6570', 'callback': 'searchresult', 'q': '圣诞树', 'stype': '1', 'pagesize': '100', 'pagenum': '1', 'imageType': '2', 'imageColor': '', 'brand': '', 'imageSType': '', 'fr': '1', 'sortFlag': '1', 'imageUType': '', 'btype': '', 'authid': '', '_': '1640268423440'}
#
# response = requests.get(url, headers=headers, params=kv)
# dict_text = json.loads(response.text[13:-1])
# imglist = dict_text['imglist']
# name = imglist[0]["caption"]
# pic_url = imglist[0]["imgurl"]
# f = open('/Users/Kelizai/PycharmProjects/Text-non-text-Classification/' + name + ".jpg", "wb")
# requests.packages.urllib3.disable_warnings()
# r_img = requests.get(pic_url, headers=headers, verify=False)
