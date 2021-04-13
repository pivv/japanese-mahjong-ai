''' Function to crawl tenhou log data.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import time
import glob

import requests
from bs4 import BeautifulSoup

def repair_logs(link_path, log_path, repaired_log_path):
    f = open(link_path, 'r')
    lines_link = f.readlines()
    f.close()
    f = open(log_path, 'r')
    lines_log = f.readlines()
    f.close()
    links = [line.split('"')[1] for line in lines_link]
    log_codes = [link.split('=')[1] for link in links]
    assert(len(lines_log) == len(log_codes))

    f = open(repaired_log_path, 'w')
    for (i, code), line_log in zip(enumerate(log_codes), lines_log):
        if(line_log.strip() != 'None'):
            f.write(line_log)
        else:
            print("%d'th log, code : %s is None, repairing."%(i, code))
            flag = True
            cnt = 0
            while(flag):
                log_text = ''
                try:
                    log_text = crawl_log_one(code)
                    assert(log_text.strip() != 'None')
                    flag = False
                except:
                    cnt += 1
                    print("Connection failed for %d times, log_text : %s."%(cnt, log_text))
                    time.sleep(3.)
            f.write(log_text)
            f.write('\n')
    f.close()

def crawl_log_one(code):
    url = 'http://e0.mjv.jp/0/log/?' + code
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'lxml')
    log_text = soup.find('mjloggm')
    return unicode(log_text)

def crawl_logs(link_path, log_path):
    f = open(link_path, 'r')
    lines = f.readlines()
    f.close()

    log_codes = [line.split('"')[1].split('=')[1] for line in lines]
    f = open(log_path, 'w')
    for i, code in enumerate(log_codes):
        print("%d'th iteration, code is : %s."%(i, code))
        flag = True
        cnt = 0
        while(flag):
            log_text = ''
            try:
                log_text = crawl_log_one(code)
                assert(log_text.strip() != 'None')
                flag = False
            except:
                cnt += 1
                print("Connection failed for %d times, log_text : %s."%(cnt, log_text))
                time.sleep(3.)
        f.write(log_text)
        f.write('\n')
    f.close()

def acquire_log_links(link_dir, link_path = None):
    dirnames = sorted(os.listdir(link_dir))
    links = []
    for dirname in dirnames:
        link_folder = os.path.join(link_dir, dirname)
        filenames = sorted(os.listdir(link_folder))
        for filename in filenames:
            if filename[0:3] == 'scc':
                link_path = os.path.join(link_folder, filename)
                f = open(link_path, 'r')
                links += f.readlines()
                f.close()
    if link_path is not None:
        f = open(link_path, 'w')
        for link in links:
            f.write(link)
        f.close()
    return links

if __name__ == "__main__":
    if len(sys.argv) == 1:
        link_dir = '../raw/tenhou/log_links/'
        link_path = '../data/log_links.txt'
        log_path = '../data/logs.txt'
    else:
        link_dir = sys.argv[1]
        link_path = sys.argv[2]
        log_path = sys.argv[3]
    print('Acquire log links.')
    acquire_log_links(link_dir, link_path)
    print('Now crawl logs.')
    crawl_logs(link_path, log_path)
