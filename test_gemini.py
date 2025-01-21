import os
import google.generativeai as genai
import pytesseract
from PIL import Image

image_str = pytesseract.image_to_string(Image.open('./data/p04770487.jpg'))



genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Transform the following text in the DwC standard: " + image_str)
print(response.text)