import requests
import json

BASE_URL = "http://localhost:8080"

def test_api():
    """Test all API endpoints"""
    
    print("📡 Testing PDF AI Repository API")
    print("=" * 60)
    
    # 1. Health check
    print("\n1. Health Check:")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 2. Upload a test PDF (you need to create or download one first)
    print("\n2. Upload PDF:")
    # First, let's create a simple text file as a test
    with open("test_document.txt", "w") as f:
        f.write("This is a test document about artificial intelligence. ")
        f.write("Machine learning is a subset of AI that enables systems to learn from data. ")
        f.write("Deep learning uses neural networks with multiple layers.")
    
    # Convert to PDF if you have a PDF, otherwise skip this test
    print("   ⚠️  Note: Need actual PDF file to test upload")
    
    # 3. Test query endpoint
    print("\n3. Test Query (without uploaded PDF):")
    query_data = {
        "question": "What is artificial intelligence?",
        "top_k": 3,
        "prompt_template": "qa"
    }
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success!")
        print(f"   Question: {result['question']}")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Sources: {len(result['sources'])}")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    # 4. Test search
    print("\n4. Test Search:")
    search_data = {
        "query": "machine learning",
        "limit": 5
    }
    response = requests.post(f"{BASE_URL}/search", json=search_data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Found {result['total_results']} results")
    else:
        print(f"   ✗ Failed: {response.text}")
    
    # 5. Test Gemini chat
    print("\n5. Test Gemini Chat:")
    chat_data = {
        "question": "Explain quantum computing in simple terms",
        "top_k": 2
    }
    response = requests.post(f"{BASE_URL}/gemini/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Conversation started")
        print(f"   Conversation ID: {result.get('conversation_id')}")
        print(f"   Answer: {result.get('answer', '')[:80]}...")
    else:
        print(f"   ✗ Failed: {response.text}")
    
    print("\n" + "=" * 60)
    print("✅ API Test Complete!")

if __name__ == "__main__":
    test_api()