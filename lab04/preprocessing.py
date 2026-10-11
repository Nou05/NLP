import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải tài nguyên NLTK (chạy lần đầu nếu chưa tải)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_html_tags(text):
    """Xóa các thẻ HTML như <br />, <p>,..."""
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def clean_text(text):
    """Hàm xử lý chuỗi văn bản hoàn chỉnh"""
    if not isinstance(text, str):
        return ""

    # 1. Xóa HTML tags
    text = remove_html_tags(text)

    # 2. Chuyển thành chữ thường và xóa ký tự không phải chữ cái
    text = re.sub(r"[^a-zA-L\s]", " ", text.lower())

    # 3. Tách từ, xóa stopwords và Lemmatize
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)