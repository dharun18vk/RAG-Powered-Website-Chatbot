import google.generativeai as genai

# 🔑 Replace with your API key
API_KEY = ""

# Configure API
genai.configure(api_key=API_KEY)

try:
    # Initialize model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Send a simple test prompt
    response = model.generate_content("Say 'API is working' if you can read this.")

    # Print response
    print("✅ API Response:")
    print(response.text)

except Exception as e:
    print("❌ Error occurred:")
    print(e)