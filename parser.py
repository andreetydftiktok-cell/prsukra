import time
import json
import os
import requests
import feedparser
from bs4 import BeautifulSoup
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()  # підхоплює .env при локальному запуску; на Railway / GitHub Actions не потрібен
except ImportError:
    pass

# ================= НАЛАШТУВАННЯ =================
BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

GROQ_API_KEYS = [k for k in os.environ.get('GROQ_API_KEYS', '').split(',') if k]

if not BOT_TOKEN or not CHAT_ID or not GROQ_API_KEYS:
    raise SystemExit(
        "❌ Не задані змінні оточення BOT_TOKEN, CHAT_ID та/або GROQ_API_KEYS. "
        "Локально: створи файл .env або задай їх у системі. На GitHub Actions: Secrets and variables → Actions."
    )

# Актуальні моделі Groq (за пріоритетом)
GROQ_MODELS = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'qwen/qwen3-32b'
]

def refresh_available_models():
    """Перевіряє, які моделі реально доступні ключу в Groq API,
    і залишає у GROQ_MODELS тільки їх."""
    global GROQ_MODELS
    try:
        models_resp = client.models.list()
        available_ids = {m.id for m in models_resp.data}
        filtered = [m for m in GROQ_MODELS if m in available_ids]
        if filtered:
            skipped = [m for m in GROQ_MODELS if m not in available_ids]
            if skipped:
                print(f"ℹ️ Цьому ключу недоступні моделі: {', '.join(skipped)} — пропускаю їх.")
            GROQ_MODELS = filtered
            print(f"✅ Буду використовувати моделі (за пріоритетом): {', '.join(GROQ_MODELS)}")
        else:
            print("⚠️ Не вдалося визначити доступні моделі, використовується список за замовчуванням.")
    except Exception as e:
        print(f"⚠️ Не вдалося отримати список моделей Groq ({e}), використовується список за замовчуванням.")

current_key_index = 0

def get_groq_client():
    return OpenAI(
        api_key=GROQ_API_KEYS[current_key_index],
        base_url="https://api.groq.com/openai/v1"
    )

client = get_groq_client()

# Повністю перевірений список робочих RSS-стрічок українських ЗМІ
RSS_URLS = [
    'https://www.pravda.com.ua/rss/',
    'https://rss.unian.net/site/news_ukr.rss',
    'https://tsn.ua/rss/full.rss',
    'https://www.rbc.ua/static/rss/all.ukr.rss.xml',
    'https://assets.censor.net/rss/censor.net/rss_uk_news.xml',
    'https://fakty.com.ua/ua/feed/',
    'https://news.liga.net/rss.xml',
    'https://interfax.com.ua/news/rss',
    'https://focus.ua/uk/rss',
    'https://suspilne.media/rss/all.rss',
    'https://sud.ua/rss/rss_news_uk.xml',
    'https://glavcom.ua/xml/rss.xml'
]

HISTORY_FILE = 'history.json'
TOPICS_FILE = 'posted_topics.json'
CHECK_INTERVAL = 300 # 5 хвилин

FIRST_RUN = True

SYSTEM_PROMPT = """Ти — топовий копірайтер із 15-річним стажем, який спеціалізується на створенні контенту для найбільших українських Telegram-каналів у стилі "Труха". 
Твоє завдання — переробити (зробити рерайт) отриманий текст (українською мовою) під формат коротких, клікабельних та динамічних новин.

Працюй за такими суворими правилами:

1. СТРУКТУРА ПОСТУ:
   - Заголовок: Головна суть новини одним динамічним реченням. Наприкінці заголовка ОБОВ'ЯЗКОВО вказуй автора або джерело інформації через тире та кому. Формат: <b>[Суть новини], — [Хто сказав / Джерело]</b>.
   - Відступ (порожній рядок).
   - Опис: Короткий, місткий рерайт решти інформації (1-2 речення). Текст має бути "живим", легким для читання, у стилістиці топових ЗМІ.

2. СУВОРЕ ОБМЕЖЕННЯ НА ФАКТИ (ЖОДНОЇ СЕБЕСТЯБАТИНИ):
   - Використовуй ТІЛЬКИ ту інформацію, яку я тобі скидаю.
   - НЕ додавай нічого від себе, не шукай інформацію в інтернеті, не додумуй контекст, не додавай старі чи супутні факти. Якщо чогось немає в моєму тексті — цього не повинно бути в твоєму результаті.
   - Спікер/автор цитати ОБОВ'ЯЗКОВО має бути в заголовку, як вказано у структурі. Якщо в тексті немає чіткого автора, пиши джерело.

3. ТЕХНІЧНЕ ПРАВИЛО:
   - Відповідай ТІЛЬКИ УКРАЇНСЬКОЮ мовою.
   - Виділяй заголовок жирним шрифтом, використовуючи ТІЛЬКИ HTML-теги <b> та </b>. Не використовуй зірочки (**) для форматування!
"""

# ================= ДOПОМІЖНІ ФУНКЦІЇ =================

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def call_groq_api(system_msg, user_msg):
    global current_key_index, client

    for model_name in GROQ_MODELS:
        attempts = 0
        max_attempts = len(GROQ_API_KEYS)

        while attempts < max_attempts:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e).lower()
                print(f"🔎 RAW ERROR ({model_name}): {e}")

                if "401" in error_str or "403" in error_str or "invalid_api_key" in error_str or "invalid api key" in error_str or "authentication" in error_str:
                    print(f"❌ Ключ №{current_key_index + 1} недійсний або заблокований! Переключаю...")
                    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                    client = get_groq_client()
                    attempts += 1
                    if attempts >= max_attempts:
                        break
                    continue
                
                elif "429" in error_str or "rate_limit" in error_str:
                    print(f"⚠️ Ліміт на API-ключі №{current_key_index + 1} вичерпано. Переключаю...")
                    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                    client = get_groq_client()
                    attempts += 1
                    continue
                
                elif "decommissioned" in error_str or "model_not_found" in error_str or "does not exist" in error_str:
                    print(f"⚠️ Модель {model_name} застаріла/не знайдена. Пробую наступну модель...")
                    break
                
                else:
                    print(f"❌ Помилка звернення до Groq API ({model_name}): {e}")
                    break

    print("❌ Жодна з моделей Groq не спрацювала!")
    return None

def is_duplicate_news(article_text, posted_topics):
    if not posted_topics:
        return False

    recent_topics = posted_topics[-15:]
    topics_list_str = "\n".join([f"- {topic}" for topic in recent_topics])
    
    dup_prompt_system = (
        "Ти — головний редактор. Твоє завдання: перевірити, чи є нова стаття точним дублікатом однієї з подій у списку.\n\n"
        "СУВОРІ ПРАВИЛА:\n"
        "1. Дублікат — це виключно та САМА подія.\n"
        "2. Збіг імен або міст — це НОВА новина!\n"
        "3. Якщо ти хоч трохи сумніваєшся — завжди обирай НОВА.\n\n"
        "Відповідай СУВОРО одним словом:\n"
        "ДУБЛІКАТ\n"
        "НОВА"
    )
    
    dup_prompt_user = f"ОПУБЛІКОВАНІ ТЕМИ:\n{topics_list_str}\n\nТЕКСТ НОВОЇ СТАТТІ:\n{article_text[:800]}"

    res = call_groq_api(dup_prompt_system, dup_prompt_user)
    
    if res and "ДУБЛІКАТ" in res.upper() and len(res) < 15:
        return True
    return False

def get_article_data(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        image_url = None
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            image_url = og_image['content']
            
        return text[:3000], image_url
    except Exception as e:
        print(f"Помилка при парсингу статті {url}: {e}")
        return "", None

def send_telegram_message(text, image_url=None):
    if image_url:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        payload = {
            'chat_id': CHAT_ID,
            'photo': image_url,
            'caption': text,
            'parse_mode': 'HTML'
        }
    else:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200 and image_url:
            print(f"Телеграм не прийняв картинку, відправляю просто текст...")
            send_telegram_message(text, image_url=None)
        elif response.status_code != 200:
            print(f"Помилка відправки в ТГ: {response.text}")
    except Exception as e:
        print(f"Помилка з'єднання з ТГ: {e}")

# ================= ОСНОВНИЙ ЦИКЛ =================

def check_news():
    global FIRST_RUN
    history = load_json(HISTORY_FILE)
    posted_topics = load_json(TOPICS_FILE)
    new_posts_found = False

    is_first_run = FIRST_RUN and len(history) == 0

    for rss_url in RSS_URLS:
        if not is_first_run:
            print(f"Перевіряю стрічку: {rss_url}")
            
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:3]:
                link = getattr(entry, 'link', None)
                title = getattr(entry, 'title', 'Без заголовка')
                
                if link and link not in history:
                    if is_first_run:
                        print(f"[Стара новина, пропускаю] {title}")
                        history.append(link)
                        new_posts_found = True
                    else:
                        print(f"🔥 Знайдено новий пост! Читаю статтю: {link}")
                        
                        article_text, image_url = get_article_data(link)
                        
                        if len(article_text) > 200: 
                            if is_duplicate_news(article_text, posted_topics):
                                print(f"🙈 Нейромережа визначила, що це ДУБЛІКАТ. Пропускаю...")
                                history.append(link)
                                new_posts_found = True
                                continue
                            
                            ai_post = call_groq_api(SYSTEM_PROMPT, article_text)
                            
                            if ai_post:
                                final_message = f"{ai_post}\n\nПост взяв на сайті:\n{link}"
                                send_telegram_message(final_message, image_url)
                                print("Пост успішно переписано і відправлено в ТГ!")
                                
                                first_line = ai_post.split('\n')[0].replace('<b>', '').replace('</b>', '')
                                posted_topics.append(first_line)
                                save_json(TOPICS_FILE, posted_topics[-40:]) 
                        
                        history.append(link)
                        new_posts_found = True
                        time.sleep(3)
        except Exception as e:
            print(f"Помилка при обробці стрічки {rss_url}: {e}")

    if new_posts_found:
        save_json(HISTORY_FILE, history[-500:])
    
    if is_first_run:
        print("\n=== Ініціалізація завершена. Тепер чекаю нові пости. ===\n")
    elif not new_posts_found:
        print("Нових постів поки немає.")
    FIRST_RUN = False

if __name__ == '__main__':
    print("AI-Парсер запущено! Збираю старі пости для бази...")
    refresh_available_models()

    if os.environ.get('RUN_ONCE', '').lower() == 'true':
        check_news()
    else:
        while True:
            try:
                check_news()
                print(f"Сплю {CHECK_INTERVAL} секунд...\n")
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                print("Зупинка скрипта...")
                break
            except Exception as e:
                print(f"Критична помилка: {e}")
                time.sleep(60)
