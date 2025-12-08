#!/usr/bin/env python
"""
Quick fix and test script for Gemini API
"""
import os
import sys
import subprocess
import json

def check_and_fix():
    print("🔧 Fixing Gemini API Issues")
    print("=" * 60)
    
    # 1. Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        
        # Check .env file
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('GEMINI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip("'\"")
                        print(f"Found API key in .env: {api_key[:10]}...")
                        break
        
        if not api_key:
            print("\n📝 Please enter your Gemini API key:")
            api_key = input("API Key: ").strip()
            
            if api_key:
                # Save to .env
                with open('.env', 'w') as f:
                    f.write(f'GEMINI_API_KEY={api_key}\n')
                    f.write('APP_ENVIRONMENT=development\n')
                    f.write('APP_DEBUG=true\n')
                print("✅ Saved to .env file")
            else:
                print("❌ No API key provided")
                return False
    
    # 2. Test the API key
    print("\n🧪 Testing API key...")
    
    test_script = """
import os
import google.generativeai as genai

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("ERROR: No API key")
    exit(1)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content("Say 'API test successful'")
    print(f"SUCCESS: {response.text}")
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
"""
    
    # Write test script
    with open('_test_gemini.py', 'w') as f:
        f.write(test_script)
    
    # Run test
    result = subprocess.run([sys.executable, '_test_gemini.py'], 
                          capture_output=True, text=True)
    
    # Clean up
    if os.path.exists('_test_gemini.py'):
        os.remove('_test_gemini.py')
    
    if result.returncode == 0:
        print(f"✅ {result.stdout.strip()}")
        return True
    else:
        print(f"❌ {result.stderr.strip()}")
        
        # Show troubleshooting tips
        print("\n🔧 Troubleshooting Tips:")
        print("1. Make sure your API key is correct")
        print("2. Try a different model: gemini-1.5-flash")
        print("3. Check internet connection")
        print("4. Get a new key: https://makersuite.google.com/app/apikey")
        
        return False

def update_config_for_gemini_2():
    """Update config for Gemini 2.0"""
    print("\n🔄 Updating configuration for Gemini 2.0...")
    
    config_file = 'config/settings.yaml'
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update model names
        content = content.replace('gemini-1.5-pro', 'gemini-2.0-flash-exp')
        content = content.replace('gemini-1.5-pro-vision', 'gemini-2.0-flash-exp')
        
        with open(config_file, 'w') as f:
            f.write(content)
        
        print("✅ Configuration updated")
    else:
        print("⚠️  Config file not found, creating basic one...")
        
        basic_config = """
app:
  name: "PDF-AI-Repository"
  version: "1.0.0"
  debug: true

paths:
  raw_pdfs: "./data/raw_pdfs/"
  vector_store: "./data/vector_store/"

gemini:
  api_key: "${GEMINI_API_KEY}"
  models:
    text: "gemini-2.0-flash-exp"
    vision: "gemini-2.0-flash-exp"
    embedding: "models/embedding-001"

database:
  vector_db_type: "simple"
  collection_name: "pdf_documents"
"""
        
        os.makedirs('config', exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(basic_config)
        
        print("✅ Basic config created")

def restart_application():
    """Restart the application"""
    print("\n🚀 Restarting application...")
    
    # Kill existing process on port 8000
    try:
        subprocess.run(['fuser', '-k', '8000/tcp'], capture_output=True)
    except:
        pass
    
    # Start the application
    print("\nStarting server...")
    print("Press Ctrl+C to stop")
    print("\nAccess points:")
    print("• API: http://localhost:8000")
    print("• Docs: http://localhost:8000/docs")
    print("• Health: http://localhost:8000/health")
    print("\n" + "=" * 60)
    
    # Start server
    subprocess.run([sys.executable, 'main.py'])

if __name__ == "__main__":
    if check_and_fix():
        update_config_for_gemini_2()
        
        print("\n" + "=" * 60)
        print("✅ Everything is ready!")
        
        response = input("\nStart the application now? (yes/no): ").strip().lower()
        if response in ['y', 'yes']:
            restart_application()
        else:
            print("\nTo start manually:")
            print("1. Run: python main.py")
            print("2. Or: uvicorn src.api.endpoints:app --reload")
            print("\nYour API key is saved in .env file")
    else:
        print("\n❌ Fix failed. Please get a valid API key.")
        print("Get free key: https://makersuite.google.com/app/apikey")