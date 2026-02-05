import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ API Key not found in .env file!")
    print("\nPlease create a .env file with:")
    print("GOOGLE_API_KEY=your_api_key_here")
else:
    print(f"✅ API Key loaded: {api_key[:20]}...")
    
    try:
        # Test with google-generativeai directly
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        print("\n📋 Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"  - {m.name}")
        
        print("\n✅ API Key is valid and working!")
        
    except Exception as e:
        print(f"\n❌ Error testing API key: {e}")