from kiteconnect import KiteConnect

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

kite = KiteConnect(api_key=API_KEY)

print("👉 Open this URL in browser:")
print(f"https://kite.trade/connect/login?api_key={API_KEY}")

request_token = input("\nPaste request_token here: ")

data = kite.generate_session(request_token, api_secret=API_SECRET)

print("\n✅ ACCESS TOKEN:")
print(data["access_token"])