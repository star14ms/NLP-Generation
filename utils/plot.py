import pandas as pd
import matplotlib.pyplot as plt


def plot_pie_lang_ratio(lang_counts: pd.Series) -> None:
    plt.figure(figsize=(10, 8))
    plt.pie(lang_counts, labels = lang_counts.index, autopct='%1.1f%%')
    plt.title('Language Ratio')
    plt.show()


def plot_pie_lang_ratio_v2(lang_counts: pd.Series) -> None:
    plt.figure(figsize=(10, 8))

    # Explode slices differently based on their proportion
    explode = []
    for value in lang_counts:
        if value < lang_counts.sum() * 0.005:
            explode.append(1)  # smaller explode for small slices
        elif value < lang_counts.sum() * 0.02:
            explode.append(0.2)  # bigger explode for larger slices
        else:
            explode.append(0)

    # Plot pie chart
    patches, texts, autotexts = plt.pie(lang_counts, labels = lang_counts.index, autopct='%1.1f%%', 
                                        textprops={'fontsize': 10}, pctdistance=0.8, explode=explode)
    
    # Increase fontsize of larger slices for better visibility
    for text, autotext in zip(texts, autotexts):
        if autotext.get_text()[:-1] and float(autotext.get_text()[:-1]) > 5:
            text.set_fontsize(12)
            autotext.set_fontsize(12)

    plt.title('Language Ratio', fontsize=14)
    plt.show()


def plot_pie_lang_ratio_v3(lang_counts: pd.Series) -> None:
    plt.figure(figsize=(20, 8))
    
    # Filter counts
    small_counts = lang_counts[lang_counts / lang_counts.sum() < 0.01]
    large_counts = lang_counts.drop(small_counts.index)
    
    # Add an 'etc' category to the large_counts
    large_counts = pd.concat([large_counts, pd.Series([small_counts.sum()], index=['Languages < 1%'])])
    
    # Plot pie chart for large counts
    plt.subplot(121)
    plt.pie(large_counts, labels = large_counts.index, autopct='%1.1f%%', pctdistance=0.8, explode=[0.1]*len(large_counts))
    plt.title('Languages', fontsize=14)

    # Plot pie chart for small counts
    plt.subplot(122)
    patches, texts, autotexts = plt.pie(small_counts, labels = small_counts.index, autopct='%1.1f%%', pctdistance=0.8, explode=[0.1]*len(small_counts))

    # Increase fontsize of larger slices for better visibility
    for text, autotext in zip(texts, autotexts):
        autotext._text = f'{round(lang_counts[text._text] / lang_counts.sum() * 100, 2)}%'

    plt.title('Breakdown of etc (< 1%)', fontsize=14)
    plt.show()


def plot_pie_lang_ratio_v4(lang_counts: pd.Series) -> None:
    plt.figure(figsize=(30, 8))
    
    # Filter counts
    smallest_counts = lang_counts[lang_counts / lang_counts.sum() < 0.001]
    small_counts = lang_counts[(lang_counts / lang_counts.sum() >= 0.001) & (lang_counts / lang_counts.sum() < 0.01)]
    large_counts = lang_counts.drop(small_counts.index.union(smallest_counts.index))
    
    # Add 'etc' categories
    large_counts = pd.concat([large_counts, pd.Series([small_counts.sum()], index=['Languages < 1%'])])
    small_counts = pd.concat([small_counts, pd.Series([smallest_counts.sum()], index=['Languages < 0.1%'])])

    # Plot pie chart for large counts
    plt.subplot(131)
    plt.pie(large_counts, labels = large_counts.index, autopct='%1.1f%%', pctdistance=0.8, explode=[0.1]*len(large_counts))
    plt.title('Languages', fontsize=14)

    # Plot pie chart for small counts
    plt.subplot(132)
    patches, texts, autotexts = plt.pie(small_counts, labels = small_counts.index, autopct='%1.1f%%', pctdistance=0.8, explode=[0.1]*len(small_counts))
    for text, autotext in zip(texts, autotexts):
        if text._text != 'Languages < 0.1%':
            autotext._text = f'{round(lang_counts[text._text] / lang_counts.sum() * 100, 2)}%'
        else:
            autotext._text = f'{round(smallest_counts.sum() / lang_counts.sum() * 100, 2)}%'
    plt.title('Breakdown of Languages < 1%', fontsize=14)

    # Plot pie chart for smallest counts
    plt.subplot(133)
    patches, texts, autotexts = plt.pie(smallest_counts, labels = smallest_counts.index, autopct='%1.1f%%', pctdistance=0.8, explode=[0.1]*len(smallest_counts))
    for text, autotext in zip(texts, autotexts):
        autotext._text = f'{round(lang_counts[text._text] / lang_counts.sum() * 100, 2)}%'
    plt.title('Breakdown of Languages < 0.1%', fontsize=14)

    # plt.savefig('lang_ratio.png')
    plt.show()
