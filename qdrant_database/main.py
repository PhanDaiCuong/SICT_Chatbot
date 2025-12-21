import os
import logging
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from qdrant_database.src.crawl import crawl_and_save_data, read_urls_from_file
from qdrant_database.src.seed_data import seed_qdrant

logging.basicConfig(level=logging.INFO)


# Load variable enviroment
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
DOC_NAME_OUTPUT = os.getenv("DOC_NAME_SICT")
DIRECTORY_DATA   = os.getenv("DIRECTORY_DATA")


if __name__=='__main__':
    CHUNK_SIZE = 2500
    CHUNK_OVERLAP = 500
    URL_FILE_PATH = "./data/link.txt"
    # Step 1: Crawl and save data 
    logging.info('Start crawl data')
    list_urls = read_urls_from_file(URL_FILE_PATH)

    print(DOC_NAME_OUTPUT)
    # Bước 2: Chạy crawl và lưu dữ liệu
    if list_urls:
        crawl_and_save_data(
            list_urls=list_urls,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            doc_name=DOC_NAME_OUTPUT,
            directory_data=DIRECTORY_DATA
        )
        logging.info('Crawl SICT data successfully!')
    else:
        logging.info("Please check your urls.txt file.")
    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        logging.info("Initialize embedding model successfully!")
    except Exception as e:
        logging.error(f"Error initializing embedding model: {str(e)}")
        embeddings = None
    
    # Step 2: send data to Qdrant database
    logging.info('Start send data')
    seed_qdrant(qdrant_host=QDRANT_HOST, qdrant_api_key=QDRANT_API_KEY, collection_name=QDRANT_COLLECTION_NAME, embedding_model=embeddings, filename=DOC_NAME_OUTPUT, directory=DIRECTORY_DATA)
    logging.info('Send send SICT data successfully!')

    logging.info('Send all data successfully!')