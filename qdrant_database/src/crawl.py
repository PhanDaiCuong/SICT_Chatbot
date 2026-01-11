import os
import requests
import time
import random 
from tqdm import tqdm 
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import uuid
import json



# --- SETUP ---
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless=new') # Chạy ẩn
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-gpu')
driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 10)


def download_image(img_url: str, save_folder):
    """Tải ảnh và trả về tên file local"""
    try: 
        # Thêm User-Agent để tránh bị server chặn request ảnh
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(img_url, headers=headers, stream=True, timeout=10)
        
        if response.status_code == 200:
            # Xử lý đuôi file
            ext = img_url.split('.')[-1].split('?')[0]
            if len(ext) > 4 or not ext: ext = "jpg"
            
            filename = f"{uuid.uuid4()}.{ext}"
            file_path = os.path.join(save_folder, filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return filename
    except Exception as e:
        # Chỉ in lỗi ngắn gọn để không làm rối màn hình
        pass 
    return None

def crawl_url(news_page_url: str, root_dir: str, images_dir: str, n_pages: int = 50, news_id: int = 0) -> None :
    
    driver.get(news_page_url)

    # 1. Định vị Main Content
    main_content_xpath = '//section[contains(@class, "irs-blog-field")]//div[@class="col-md-8"]'
    try:
        main_content_tag = wait.until(EC.presence_of_element_located((By.XPATH, main_content_xpath)))
    except:
        return f"Không tìm thấy nội dung chính tại {news_page_url}"
    
    
    # --- TRÍCH XUẤT DỮ LIỆU ---

    # 2. Lấy Title
    title = ""
    try:
        title = main_content_tag.find_element(By.XPATH, './/p[@class="pTitle"]').text.strip()
    except: pass

    # 3. Lấy Abstract
    abstract = ""
    try:
        abstract = main_content_tag.find_element(By.XPATH, './/p[@class="pHead"]').text.strip()
    except: pass

    # 4. Lấy Body Text VÀ Hình ảnh
    content_text = ""
    images_data = []
    
    try:
        # [SỬA LỖI] Dùng CSS Selector để lấy p, h2, h3 cùng lúc
        # Dấu phẩy nghĩa là "hoặc" (lấy p HOẶC h2 HOẶC h3)
        content_tags = main_content_tag.find_elements(By.CSS_SELECTOR, "p, h2, h3, ul, tr")
        
        text_parts = []
        
        for tag in content_tags:
            # --- KIỂM TRA ĐỂ TRÁNH LẶP ---
            tag_class = tag.get_attribute("class")
            
            if tag_class and ("pTitle" in tag_class or "pHead" in tag_class):
                continue
            # 1. Lấy Text và Format theo thẻ
            text = tag.text.strip()
            if text:
                # Nếu là Header thì thêm dấu # để file txt đẹp hơn (Markdown style)
                if tag.tag_name == 'h2':
                    text_parts.append(f"\n## {text}")
                elif tag.tag_name == 'h3':
                    text_parts.append(f"\n### {text}")
                else:
                    text_parts.append(text)
            
            # 2. Lấy Ảnh (tìm trong tag hiện tại)
            imgs_in_tag = tag.find_elements(By.TAG_NAME, "img")
            for img in imgs_in_tag:
                src = img.get_attribute('src')
                if src:
                    saved_filename = download_image(src, images_dir)
                    if saved_filename:
                        images_data.append({
                            "original_url": src,
                            "local_filename": saved_filename,
                            "relative_path": f"images/{saved_filename}"
                        })

        content_text = '\n'.join(text_parts)

    except Exception as e:
        print(f"Lỗi parse body: {e}")

    # --- LƯU JSON ---
    article_data = {
        "id": f"ttc_{news_id:03d}",
        "url": news_page_url,
        "title": title,
        "abstract": abstract,
        "content": content_text,
        "images": images_data 
    }
    
    # --- LƯU FILE TXT ---
    final_content_lst = [title.upper(), abstract, content_text]
    final_content = '\n\n'.join([x for x in final_content_lst if x]) # Lọc bỏ phần tử rỗng

    news_filename_json = f"news_sict_{news_id:03d}.json"
    news_savepath_json = os.path.join(root_dir, news_filename_json)
    
    news_filename_txt = f"news_sict_{news_id:03d}.txt"
    news_savepath_txt = os.path.join(root_dir, news_filename_txt)
    
    with open(news_savepath_json, 'w', encoding='utf-8') as f:
        json.dump(article_data, f, ensure_ascii=False, indent=4)
    
    with open(news_savepath_txt, 'w', encoding='utf-8') as f:
        f.write(final_content)

    return f"Đã lưu xong: {news_filename_json} và {news_filename_txt}"

if __name__ == "__main__":
    
    list_schools = ["sict", "seee", "smae"]
    for school in list_schools:
        # Cấu hình đường dẫn
        root_dir = f'./{school}_corpus/daotao/saudaihoc/thacsi/codientu'
        images_dir = os.path.join(root_dir, 'images')
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        n_pages = 50
        news_id = 0
        
        for page_idx in tqdm(range(1, n_pages + 1), desc="Pages"):
            try:
                # 1. Vào trang danh sách
                main_url = f'https://{school}.haui.edu.vn/vn/cong-trinh-khcn/{page_idx}'
                driver.get(main_url)# Lấy danh sách link (Dùng wait để đảm bảo list đã load)
                news_lst_xpath = '//section[contains(@class, "irs-blog-field left-img irs-blog-single-field")]//h2/a'
                wait.until(EC.presence_of_element_located((By.XPATH, news_lst_xpath)))
                
                news_tags = driver.find_elements(By.XPATH, news_lst_xpath)
                # Lưu lại list URL để tránh lỗi StaleElement khi chuyển trang
                news_page_urls = [tag.get_attribute('href') for tag in news_tags]
                for url in news_page_urls:
                    crawl_url(url, root_dir, images_dir, n_pages, news_id)
            except Exception as e:
                print(f"Lỗi trang chính danh sách url{page_idx}: {e}")