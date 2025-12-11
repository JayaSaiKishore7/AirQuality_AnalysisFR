import requests
import time

print("=" * 50)
print("🧪 Testing FastAPI Connection")
print("=" * 50)

API_URL = "http://127.0.0.1:8000"

# Test 1: Basic connection
print("\n1. Testing basic connection...")
try:
    start = time.time()
    response = requests.get(f"{API_URL}/", timeout=5)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        print(f"   ✅ SUCCESS: Status {response.status_code}")
        print(f"   Response: {response.json()}")
        print(f"   Response time: {elapsed:.2f}s")
    else:
        print(f"   ❌ FAILED: Status {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"   ❌ ERROR: {type(e).__name__}: {e}")

# Test 2: Metadata
print("\n2. Testing metadata endpoint...")
try:
    start = time.time()
    response = requests.get(f"{API_URL}/meta", timeout=10)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ SUCCESS: Status {response.status_code}")
        print(f"   Pollutants: {len(data.get('pollutants', []))}")
        print(f"   Sites: {len(data.get('sites_sample', []))}")
        print(f"   Response time: {elapsed:.2f}s")
    else:
        print(f"   ❌ FAILED: Status {response.status_code}")
        
except Exception as e:
    print(f"   ❌ ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("✅ Test complete!")