
#API_KEY = "AIzaSyDMgfn0CCGL9c3Zb6LO6Gcxt2IIl3XR1Nc"

from google import genai

# Test prompt
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what RAG is in simple terms"
)

print(response.text)