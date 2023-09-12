import requests
from bs4 import BeautifulSoup


def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text_data = [element.get_text() for element in soup.find_all(['body'])]
    return " ".join(text_data)


def clean_and_split_data(data):
    split_data = data.split('\n')
    clean_data = [info.strip() for info in split_data if info.strip()]
    return clean_data
