from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import matplotlib.pyplot as plt
from pycountry import languages

# from memory_profiler import profile
# from rich import print
from utils._rich import new_progress
from utils.plot import plot_pie_lang_ratio_v4


def read_data(data_path: str) -> list[str]:
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return lines


def detect_language(texts: list[str], dest_path: str) -> list[dict]:
    len_texts = len(texts)
    task_id = progress.add_task(f'Detecting language (0/{len_texts})', total=len_texts)
    result = []

    with open(dest_path, 'w', encoding='utf-8') as f:
        for idx, text in enumerate(texts):
            text = text.strip()
            if text:
                try:
                    detected_lang = detect(text)
                except LangDetectException:
                    detected_lang = 'unknown'
                except:
                    breakpoint()
    
                result.append({
                    'id': idx,
                    'lang': detected_lang,
                    'text': text,
                })
    
                if detected_lang == 'en':
                    f.write(text + '\n')

            progress.log(detected_lang, text)
            progress.update(
                task_id=task_id,
                advance=1,
                description=f'Detecting language ({idx+1}/{len_texts})',
                refresh=True,
            )

    return result


def get_language_ratio(texts: list[str]) -> dict[str, int]:
    len_texts = len(texts)
    task_id = progress.add_task(f'Detecting language (0/{len_texts})', total=len_texts)
    result = {}

    for idx, text in enumerate(texts):
        text = text.strip()
        if text:
            detected_lang = detect(text)
            if detected_lang in result:
                result[detected_lang] += 1
            else:
                result[detected_lang] = 1

        progress.log(detected_lang, text)
        progress.update(
            task_id=task_id,
            advance=1,
            description=f'Detecting language ({idx+1}/{len_texts})',
            refresh=True,
        )

    return result


def save_data_to_csv(data: list[dict], save_path: str) -> None:
    '''
    data = [
        {'id': 0, 'lang': 'en', 'text': '...'},
        # Add more dictionaries for each additional row
        # ...
    ]
    '''

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Write DataFrame to CSV file
    df.to_csv(save_path, index=False)


def iso639_1_to_name(code: str) -> str:
    lang = languages.get(alpha_2=code)

    if lang is None:
        return 'unknown'
    return languages.get(alpha_2=code).name


def language_analysis(data_path: str) -> None:
    # Read data from CSV file
    df = pd.read_csv(data_path)

    # Convert ISO 639-1 codes to full language names
    df['lang'] = df['lang'].map(iso639_1_to_name)

    # Count occurrences of each language
    lang_counts = df['lang'].value_counts()

    # Plot pie chart
    plot_pie_lang_ratio_v4(lang_counts)


if __name__ == '__main__':
    file_path = 'data/yt_cmts_230624.txt'
    dest_path = 'data/yt_cmts_230624_en.txt'
    save_path = 'data/yt_cmts_230624.csv'

    progress = new_progress()
    progress.start()
    
    texts = read_data(file_path)
    result = detect_language(texts, dest_path)
    # save_data_to_csv(result, save_path)

    # language_analysis(save_path)
