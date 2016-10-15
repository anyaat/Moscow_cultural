import matplotlib
import pandas as pd
import seaborn as sns


# если в ipython notebook, раскоментируйте следующую строчку
# %matplotlib inline


def get_price_category(df):
    """
    Создаем data frame c количеством мероприятий каждой ценовой категории для каждой категории событий.  
    """
    price_gr = []

    # копируем исходный data frame и добавляем новый столбец в новый data frame
    for row in mc['price']:
        if row == 'free':
            price_gr.append('free')
        elif float(row) <= 100:
            price_gr.append('< 100')
        elif float(row) <= 500:
            price_gr.append('< 500')
        elif float(row) <= 1000:
            price_gr.append('< 1000')
        elif float(row) <= 5000:
            price_gr.append('< 5000')
        elif float(row) <= 64000:
            price_gr.append('< 63000')
        else:
            price_gr.append(row)
    dfExt = mc
    dfExt['price_gr'] = price_gr

    # оставляемы только интересующие нас колонки и удаляем строки, в которых значение одной или значения обеих 
    # категорий не определены
    dfExt = dfExt.ix[:,['category','price_gr']].dropna()

    # изменяем имена категорий на английские
    for rus, eng in zip(['выставки', 'кинопоказы', 'концерты', 'лекции мк', 'спектакли', 'фестивали'], 
                        ['exhibitions', 'films', 'concerts', 'lectures', 'shows', 'festivals']):
        dfExt.category = dfExt.category.map(lambda x: x.replace(rus, eng))

    exh = dfExt[dfExt['category'] == 'exhibitions']
    films = dfExt[dfExt['category'] == 'films']
    conc = dfExt[dfExt['category'] == 'concerts']
    lect = dfExt[dfExt['category'] == 'lectures']
    plays = dfExt[dfExt['category'] == 'shows']
    fest = dfExt[dfExt['category'] == 'festivals']

    fn = pd.DataFrame({'category':['festivals', 'festivals', 'festivals', 'festivals', 'festivals', 'festivals']})
    fn['price'] = ['< 100', '< 500', '< 1000', '< 5000', '< 63000', 'free']
    fn['count'] = [i/466 for i in [21, 197, 84, 57, 5, 102]]
    pn = pd.DataFrame({'category':['plays', 'plays', 'plays', 'plays', 'plays', 'plays']})
    pn['price'] = ['< 100', '< 500', '< 1000', '< 5000', '< 63000', 'free']
    pn['count'] = [i/694 for i in [6, 116, 223, 315, 16, 18]]
    ln = pd.DataFrame({'category':['lectures', 'lectures', 'lectures', 'lectures', 'lectures', 'lectures']})
    ln['price'] = ['< 100', '< 500', '< 1000', '< 5000', '< 63000', 'free']
    ln['count'] = [i/942 for i in [25, 431, 116, 132, 72, 166]]
    cn = pd.DataFrame({'category':['concerts', 'concerts', 'concerts', 'concerts', 'concerts', 'concerts']})
    cn['price'] = ['< 100', '< 500', '< 1000', '< 5000', '< 63000', 'free']
    cn['count'] = [i/515 for i in [10, 231, 139, 106, 12, 17]]
    en = pd.DataFrame({'category':['exhibitions', 'exhibitions', 'exhibitions', 'exhibitions', 'exhibitions', 'exhibitions']})
    en['price'] = ['< 100', '< 500', '< 1000', '< 5000', '< 63000', 'free']
    en['count'] = [i/1128 for i in [243, 757, 22, 1, 0, 105]]
    filn = pd.DataFrame({'category':['films', 'films', 'films', 'films', 'films', 'films']})
    filn['price'] = ['< 100', '< 500', '< 1000', '< 5000', '< 63000', 'free']
    filn['count'] = [i/328 for i in [20, 142, 84, 23, 0, 59]]

    mergedNorm = pd.concat([fn,pn,ln,cn,en,filn])

    return mergedNorm

def main():
    sns.set_palette("deep", desat=.6)
    sns.set(font_scale=1.7)
    sns.set_context(rc={"figure.figsize": (17, 7)})
    mc = pd.read_csv('cultmos.csv', sep='\t')
    
    sns.barplot(x="category", y="count", hue="price", data=get_price_category(mc))

if __name__ == "__main__":
    main()
