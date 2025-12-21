import os
import re
import json
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import logging
from typing import List



LOGGER = logging.getLogger(__name__)

load_dotenv()


def bs4_extractor(html: str) -> str:
    """
    Chỉ trích xuất nội dung từ các class: pTitle, pHead, pBody, pull-left
    """
    soup = BeautifulSoup(html, "html.parser")
    
    target_selector = ".pTitle, .pHead, .pBody, .pull-left"
    
    elements = soup.select(target_selector)
    
    # Trích xuất text từ các thẻ tìm được
    text_parts = []
    for el in elements:
        # get_text(separator=' ', strip=True) giúp làm sạch text bên trong thẻ luôn
        text_content = el.get_text(separator=' ', strip=True)
        if text_content: # Chỉ lấy nếu có nội dung
            text_parts.append(text_content)
            
    # Nối các phần lại với nhau bằng 2 dấu xuống dòng để tách đoạn
    return "\n\n".join(text_parts)


def crawl_web(url_data: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Crawl data from a URL recursively
    Args:
        url_data (str): Root URL to start crawling
    Returns:
        list: List of Document objects, each object containing split content
              and corresponding metadata
    """
    # Create a loader with a maximum depth of 4 levels
    loader = RecursiveUrlLoader(url=url_data, extractor=bs4_extractor, max_depth=4)
    docs = loader.load()  # Load content
    print('length: ', len(docs))  # Print the number of loaded documents

    # Split text into chunks of 10000 characters, with 500 character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)
    print('length_all_splits: ', len(all_splits))  # Print the number of text chunks after splitting
    return all_splits

def web_base_loader(url_data: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Load data from a single URL (non-recursive)
    Args:
        url_data (str): URL to load data from
    Returns:
        list: List of split Document objects
    """
    loader = WebBaseLoader(url_data)  # Create a basic loader
    docs = loader.load()  # Load content
    print('length: ', len(docs))  # Print the number of documents

    # Split text similarly to above
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def save_data_locally(documents, filename, directory):
    """
    Save a list of documents to a JSON file
    Args:
        documents (list): List of Document objects to save
        filename (str): JSON filename (e.g., 'data.json')
        directory (str): Directory path to save the file
    Returns:
        None: This function does not return a value, it only saves the file and prints a message
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)  # Create the full file path

    # Convert documents to a serializable format
    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    # Save to JSON file
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print(f'Data saved to {file_path}')  # Print a success message


def save_data_locally(documents, filename, directory):
    """Save documents to JSON"""
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    
    # Thêm encoding='utf-8' và ensure_ascii=False để lưu tiếng Việt không bị lỗi font
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_to_save, file, indent=4, ensure_ascii=False)
    
    LOGGER.info(f'Data saved to {file_path}')


def read_urls_from_file(file_path: str) -> List[str]:
    """
    Đọc danh sách URL từ file text.
    Bỏ qua dòng trống và dòng bắt đầu bằng dấu #.
    """
    urls = []
    if not os.path.exists(file_path):
        LOGGER.error(f"File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith('#'):
                urls.append(clean_line)
    
    LOGGER.info(f"Found {len(urls)} URLs in {file_path}")
    return urls

def crawl_and_save_data(list_urls: List[str], chunk_size: int, chunk_overlap: int, doc_name: str, directory_data: str) -> None:
    """
    Hàm điều phối chính: Duyệt qua list URL -> Crawl -> Gộp data -> Lưu
    """
    all_documents = []

    for url in list_urls:
        docs = crawl_web(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_documents.extend(docs)

    if all_documents:
        LOGGER.info(f"Total documents collected: {len(all_documents)}")
        if directory_data is None:
            directory_data = "./data"
        save_data_locally(all_documents, doc_name, directory_data)
        LOGGER.info("All processing complete!")
    else:
        LOGGER.warning("No data collected from any URL.")