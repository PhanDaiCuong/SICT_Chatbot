import os
import uuid
import requests
import streamlit as st

# Lấy URL từ biến môi trường. 
# Lưu ý: Trong docker-compose, biến này nên là 'http://chatbot_api:8080/chatbot-rag-agent'
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8080/chatbot-rag-agent")

# Tạo user_id tự động cho mỗi phiên làm việc
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# --- CẤU HÌNH SIDEBAR (THANH BÊN) ---
with st.sidebar:
    st.header("Giới thiệu")
    st.markdown(
        """
        Đây là **SICT News Chatbot** - Trợ lý ảo hỗ trợ giải đáp thắc mắc về:
        
        * 📰 **Tin tức & Sự kiện** nhà trường.
        * 🎓 **Thông tin Đào tạo** (Lịch thi, thời khóa biểu).
        * 📢 **Tuyển sinh** và Công tác sinh viên.
        
        Hệ thống sử dụng công nghệ **RAG (Retrieval-Augmented Generation)** để tìm kiếm thông tin chính xác nhất từ cơ sở dữ liệu của trường SICT.
        """
    )

    st.header("Câu hỏi gợi ý")
    st.markdown("- Học phí ngành Công nghệ thông tin năm nay là bao nhiêu?")
    st.markdown("- Sắp tới trường có sự kiện gì cho tân sinh viên không?")
    st.markdown("- Điều kiện để đạt học bổng khuyến khích học tập?")
    st.markdown("- Thời gian đăng ký tín chỉ học kỳ này?")
    st.markdown("- Thủ tục xin giấy xác nhận sinh viên như thế nào?")
    st.markdown("- Điểm chuẩn xét tuyển năm ngoái là bao nhiêu?")
    st.markdown("- Quy định về trang phục khi đến trường?")
    st.markdown("- Liên hệ phòng Công tác sinh viên ở đâu?")

# --- GIAO DIỆN CHÍNH ---
st.title("🏛️ HỆ THỐNG HỎI ĐÁP TIN TỨC SICT")
st.caption("Trường Công nghệ Thông tin và Truyền thông")

st.info(
    "👋 Chào bạn! Hãy hỏi tôi bất cứ điều gì về lịch học, học phí, sự kiện hoặc quy chế của trường SICT."
)

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["output"])

# --- XỬ LÝ NHẬP LIỆU ---
if prompt := st.chat_input("Bạn đang quan tâm đến thông tin gì?"):
    # Hiển thị câu hỏi của người dùng ngay lập tức
    st.chat_message("user").markdown(prompt)

    # Lưu câu hỏi vào session
    st.session_state.messages.append({"role": "user", "output": prompt})

    # Chuẩn bị dữ liệu gửi đi
    data = {
        "user_id": st.session_state.user_id,
        "message": prompt
    }

    # Gọi API Backend
    with st.spinner("Đang tra cứu thông tin trường SICT..."):
        try:
            # Lưu ý: CHATBOT_URL phải đúng địa chỉ của container API
            response = requests.post(CHATBOT_URL, json=data)

            if response.status_code == 200:
                response_data = response.json()
                # Lấy nội dung trả lời từ key 'response' (hoặc key khác tùy backend của bạn)
                output_text = response_data.get("response", "Xin lỗi, tôi không tìm thấy câu trả lời.")
            else:
                output_text = (
                    f"Lỗi kết nối đến máy chủ (Mã lỗi: {response.status_code}). "
                    "Vui lòng thử lại sau."
                )

        except requests.exceptions.RequestException as e:
            output_text = f"Không thể kết nối đến Backend Chatbot. Chi tiết lỗi: {e}"

    # Hiển thị phản hồi từ bot
    st.chat_message("assistant").markdown(output_text)

    # Lưu phản hồi của bot vào lịch sử
    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
        }
    )