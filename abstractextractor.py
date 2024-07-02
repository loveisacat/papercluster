import requests
import PyPDF2
import re

import os
import xml.etree.ElementTree as ET
import pandas as pd

import requests
from bs4 import BeautifulSoup
import pyhttpx

def extract_abstract_content(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the 'abstract' element
    abstract_element = root.find('.//abstract')  # Adjust the path if the abstract is nested deeper
    title_element = root.find('.//article-title')  # Adjust the path if the abstract is nested deeper

    # Extract content from the 'abstract' element
    if abstract_element is not None:
        # Get all text recursively from the abstract element
        abstract_text = ''.join(abstract_element.itertext()).strip()
        title_text = ''.join(title_element.itertext()).strip()
        return abstract_text,title_text
    else:
        return "Abstract not found","Title not found"

# Assuming the path to the XML file
# file_path = 'Early.cermxml'
# abstract_text, title = extract_abstract_content(file_path)
# print(title)
# print("Extracted Abstract:", abstract_text)

def parse_xml_files(directory):
    data = {'title': [], 'abstract': []}

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.cermxml'):
            file_path = os.path.join(directory, filename)
            # Extract the abstract content
            abstract_content, title = extract_abstract_content(file_path)
            # Append results to the data dictionary
            data['title'].append(title)
            data['abstract'].append(abstract_content)
            
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    return df


def extract_abstract_from_pdf(file_path):
    try:
        # Open the PDF file
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Read the first page
            first_page = reader.pages[0]
            text = first_page.extract_text()
            
            if not text:
                return "No text could be extracted from the first page."
            
            # Regular expression to find the abstract
            # Adjust the regex according to the common structure of your documents            
            abstract_match = re.search(r'(?i)abstract\s*(.*?)(?=\n\s*\w+\s*:)', text, re.DOTALL)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
            else:
                abstract = "Abstract not found"
            return abstract
            
    except Exception as e:
        return f"An error occurred: {e}"


def get_abstract_from_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        abstract = data['message'].get('abstract')
        if abstract:
            return abstract
        else:
            return "Abstract not available."
    else:
        return f"Error: Unable to fetch data (status code: {response.status_code})"
    


def get_abstract_from_google_scholar(doi):
    # Replace spaces with plus for URL encoding
    query = doi.replace('http://dx.doi.org/10','10')
    query = '+'.join(query.split())
    
    # Form the URL to search the query on Google Scholar
    url = f'https://scholar.google.com/scholar?q={query}'
    #print(url)
    # # Send the request
    # response = requests.get(url)
    # print(response.text)  
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    }

    session = pyhttpx.HttpSession()
    response = session.get(url,headers=headers)
    content = response.text
    
    #print(content)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the element containing the abstract
        # This is a simplified example; you'll need to adjust the selectors based on actual page structure
        abstract_tag = soup.find('div', class_='gsh_csp')
        if abstract_tag:
            return abstract_tag.text
        else:
            return "Abstract not found."
    else:
        return "Failed to retrieve information."
    
def get_abstracts_from_cermine(directory):
    # Specify the directory containing the XML files
    df = parse_xml_files(directory)
    #print(df.head())  # Print the first few rows of the DataFrame
    return df


def check_abstract(content):
    if isinstance(content, str):
            if len(content) < 50:
                return False
    else:
            return False
    return True


def get_abstracts(file, directory):
    df_pdf = get_abstracts_from_cermine(directory)
    df_list = pd.read_excel(file)
    df_merged = pd.merge(df_list, df_pdf, on='title', how='left')
    for index, row in df_merged.iterrows():
        abstract = row['abstract']
        doi = row['DOI'].replace('http://dx.doi.org/10','10')
        if not check_abstract(abstract):
            df_merged.at[index, 'abstract'] = get_abstract_from_doi(doi)
        if not check_abstract(row['abstract']):
            df_merged.at[index, 'abstract'] = get_abstract_from_google_scholar(doi)
    df_merged.to_csv('paper.csv', index=False)
    return df_merged
