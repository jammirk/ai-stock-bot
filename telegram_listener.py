import requests
import time

BOT_TOKEN = "YOUR_TOKEN"

def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    return requests.get(url, params=params).json()

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": text})

offset = None

print("Listening for button clicks...")

while True:
    data = get_updates(offset)

    if "result" in data:
        for update in data["result"]:
            offset = update["update_id"] + 1

            if "callback_query" in update:
                query = update["callback_query"]
                chat_id = query["message"]["chat"]["id"]
                data_text = query["data"]

                parts = data_text.split("|")

                if parts[0] == "BUY":
                    stock = parts[1]
                    entry = parts[2]

                    send_message(chat_id, f"✅ BUY CONFIRMED: {stock} @ {entry}")

                    # 👉 Later: place order here

                elif parts[0] == "SKIP":
                    stock = parts[1]
                    send_message(chat_id, f"❌ SKIPPED: {stock}")

    time.sleep(2)