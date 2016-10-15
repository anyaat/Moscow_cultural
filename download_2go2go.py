import requests
import itertools
import csv
import os
import re
import json
from lxml import html


def rem ():
    '''
    removes event_info.csv if already exists
    '''
    try:
        os.remove('event_info.csv')
    except OSError:
        pass


def hrefs (d,m,y,pg):
     '''
    finds all event links for a given date and page number
    input - d: current day
            m: current month
            y: current year
            pg: current page

    output - {date: links} - dictionary where key is the current date of a page, and the value is a list of all event links for that date.
    '''
    site = requests.get('http://www.2do2go.ru/msk/events?date=%s-%s-%s&page=%s'%(d, m, y, pg))
    tree = html.fromstring(site.content)
    links = tree.xpath('//a[@class="medium-events-list_link"]/@href')
    date = "%s.%s.%s"%(d, m, y)
    return {date: links}


def max_pg(d,m,y,pg):
    '''
    finds max page of given date combination
    input - d: day
            m: month
            y: year
            pg: current page
    output - max_p - max page value for given date combination
    '''    
    site = requests.get('http://www.2do2go.ru/msk/events?date=%s-%s-%s&page=%s'%(d, m, y, pg))
    tree = html.fromstring(site.content)
    if tree.xpath('//a[@class="paginator_link"]'):
        max_p = tree.xpath('//a[@class="paginator_link"]/text()')[-2]
    else:
        max_p = None
    return max_p


def incr (d,m,y,pg,max_p):
    '''
    Increments the day, month, and years in order to crawl through and save dates of all event pages of site.
    input - d: day
            m: month
            y: year
            pg: current page
            max_p: max page number for a date combination
    output - d,m,y,pg - returns adjusted d,m,y, and pg values
    '''
    try:
        max_p, pg = int(max_p), int(pg)
    except TypeError:
        pass
    if max_p is pg or max_p is None:
        if m in (1, 3, 5, 7, 9, 11):
            if d == 31:
                d = 0
                m += 1
        elif m == 2:
            if (y == 2016 and d == 29) or d == 28:
                d = 0
                m += 1
        else:
            if d == 30:
                d = 0
                if m == 12:
                    y += 1
                    m = 1
                else:
                    m += 1
        d += 1
        pg = 1
    else:
        pg += 1
    return (d,m,y,pg)


def extract (href_list):
    '''
    Extracts information from the links stored in href_list and stored them into values, which are then sent to csv function
    input - href_list - imported json file containg links to event pages
    output - link, text, event, loc, st, date, cat, price, first - values holding coresponding information from text
    '''
    done = {}
    print ("Beginning information extraction...")
    first = False
    for dic in href_list:
        for date, link_list in dic.items():
            for link in link_list:
                if link not in done:
                    done[link] = [date]
                else:
                    done[link].append(date)
    for link,v in done.items():
        date = v[0:len(v)]
        date = ', '.join(date)
        eventpg = requests.get(link)
        tree = html.fromstring(eventpg.content.decode('utf-8'))
        text = tree.xpath('.//p/text() | .//p/strong/text() | .//p/em/text() | .//p/span/text() | .//p/a/text()')
        text = text[0:len(text) - 4]
        text = ' '. join(text)
        clean_text = re.sub(r'((\', \')| ,|\'|\\ x a 0|\\n|\[|\]|\||)',r'', text) 
        event = tree.xpath('.//h1[@itemprop="name"]/text()')[0]
        try:
            loc = tree.xpath('//a[@class="event-schedule_place-link"]/text()')[0]
        except IndexError:
            loc = 'N/A'
        st = tree.xpath('//div[@class="event-schedule_street"]/text()')[0]
        try:
            dur = tree.xpath('//div[@class="event-schedule_duration"]/text()')[0]
            dur = re.sub(r'Длительность:',r'', dur)
        except IndexError:
            dur = 'N/A'
        cat = tree.xpath('//div[@class="event-info_labeled js-categories-list"]/.//a/text()')
        cat = cat[0:len(cat)]
        cat = ' '.join(cat)
        price = tree.xpath('//div[@class="form-inline"]/text()')[0]
        views = tree.xpath('//div[@class="counters_item counters_item__views"]/text()')[0]
        comm = tree.xpath('//a[@class="counters_item counters_item__comments"]/text()')[0]
        csv_write(link, clean_text, event, loc, st, date, dur, cat, price, views, comm, first)
        first = True


def csv_write (link, text, event, loc, st, date, dur, cat, price, views, comm, first):
    '''
    writes imported info line to csv file
    input - link, text, event, loc, st, date, dur, cat, price, views, comm, first - values holding coresponding information
    output - event_info.csv
    '''
    first_row = ['link', 'text', 'event', 'location', 'street', 'date', 'duration', 'category', 'views', 'comments']
    info_row = [link, text, event, loc, st, date, dur, cat, views, comm]
    with open('event_info.csv', 'a', encoding = 'utf-8') as csv_f:
        csv_file = csv.writer(csv_f, delimiter = ';')
        if first == False:
            csv_file.writerow(first_row)
        csv_file.writerow(info_row)


rem()                               #removes prior csvfile if it exists in folder.
print ("Prior csv file removed.")
d = 1                               #sets day, month, year, and page values.
m = 1
y = 2013
pg = 1
href_list = []
print ("Beginning href extraction...")
while y <= 2015 or (m < 6 and y <= 2016):   #extracts pages until the 01.06.2016.
    href_list.append(hrefs(d,m,y,pg))       #href_list is appended by results of hrefs function.
    if pg == 1:                             #if page is first page then program finds the max amount of pages for one date.
        max_p = max_pg(d,m,y,pg)
    print ("%s.%s.%s   pg:%s"%(d,m,y,pg))   #to identify progess of program. Unnecessary. 
    incr_tup = incr(d,m,y,pg, max_p)        #date is incremented.
    d = incr_tup[0]
    m = incr_tup[1]
    y = incr_tup[2]
    pg = incr_tup[3]
with open('links1.json', 'w') as hrefs:     #Due to the length of the program, in case it is interrupted, 
    hrefs.write(json.dumps(href_list))      #href_list is printed to json format, so it can be started from the following line.
with open('links1.json', 'r') as hrefs:     #afterwards the file is read back in
    href_list = hrefs.read()
    href_list = json.loads(href_list)
extract(href_list)                          #info is extracted for each link and written to a csv file.
print ("Program Complete.")
