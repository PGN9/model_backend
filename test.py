import requests
import json
import time

# Test configuration
BACKEND_URL = "http://localhost:8000"  # Change this to your Render URL when deployed
# BACKEND_URL = "https://your-render-app.onrender.com"  # Use this for deployed version

# Test data with diverse comments covering different topics
test_comments = [
    {
        "id": "test1",
        "body": "I think the new iPhone 15 has amazing camera quality and the battery life is much better than previous models."
    },
    {
        "id": "test2", 
        "body": "My doctor told me to take vitamin D supplements during winter months for better immune system support."
    },
    {
        "id": "test3",
        "body": "The Lakers game last night was incredible! LeBron's performance in the fourth quarter was legendary."
    },
    {
        "id": "test4",
        "body": "I'm planning a trip to Japan next spring. Any recommendations for must-visit places in Tokyo?"
    },
    {
        "id": "test5",
        "body": "This new tax policy is going to hurt small businesses. The government needs to reconsider this decision."
    },
    {
        "id": "test6",
        "body": "Just finished watching the latest Marvel movie. The special effects were mind-blowing!"
    },
    {
        "id": "test7",
        "body": "I tried this new pasta recipe with truffle oil and it was absolutely delicious. Highly recommend!"
    },
    {
        "id": "test8",
        "body": "My relationship has been going through a rough patch lately. Not sure how to fix things."
    },
    {
        "id": "test9",
        "body": "Scientists discovered a new exoplanet that might have conditions suitable for life. This is fascinating!"
    },
    {
        "id": "test10",
        "body": "Can someone explain how machine learning algorithms actually work? I'm completely lost."
    }
]

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=30)
        print(f"✅ Health check status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Model: {health_data.get('model', 'Unknown')}")
            print(f"   Device: {health_data.get('device', 'Unknown')}")
            print(f"   Available topics: {len(health_data.get('available_topics', []))}")
        else:
            print(f"❌ Health check failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=30)
        print(f"✅ Root endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Message: {response.json().get('message', 'No message')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_small_batch():
    """Test with 3 comments first"""
    print("\n🔍 Testing small batch (3 comments)...")
    small_batch = test_comments[:3]
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"comments": small_batch},
            timeout=120  # 2 minutes timeout
        )
        end_time = time.time()
        
        print(f"✅ Small batch status: {response.status_code}")
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Model used: {data.get('model_used', 'Unknown')}")
            print(f"   Comments processed: {data.get('comments_processed', 0)}")
            print(f"   Memory usage: {data.get('memory_initial_mb', 0):.1f} -> {data.get('memory_peak_mb', 0):.1f} MB")
            
            # Show results for each comment
            for result in data.get('results', [])[:3]:  # Show first 3
                print(f"\n   📝 Comment {result.get('id', 'Unknown')}:")
                print(f"      Text: {result.get('body', '')[:60]}...")
                print(f"      Topic: {result.get('topic', 'Unknown')} (score: {result.get('topic_score', 0):.3f})")
                
                # Show top 3 topics
                all_scores = result.get('all_topic_scores', {})
                top_3 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"      Top 3: {', '.join([f'{topic}({score:.3f})' for topic, score in top_3])}")
        else:
            print(f"❌ Small batch failed: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Small batch error: {e}")
        return False

def test_full_batch():
    """Test with all 10 comments"""
    print("\n🔍 Testing full batch (10 comments)...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"comments": test_comments},
            timeout=300  # 5 minutes timeout
        )
        end_time = time.time()
        
        print(f"✅ Full batch status: {response.status_code}")
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Model used: {data.get('model_used', 'Unknown')}")
            print(f"   Comments processed: {data.get('comments_processed', 0)}")
            print(f"   Memory usage: {data.get('memory_initial_mb', 0):.1f} -> {data.get('memory_peak_mb', 0):.1f} MB")
            print(f"   Data size: {data.get('total_data_size_kb', 0):.1f} KB -> {data.get('total_return_size_kb', 0):.1f} KB")
            
            # Count topics
            topic_counts = {}
            for result in data.get('results', []):
                topic = result.get('topic', 'unknown')
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            print(f"\n   📊 Topic distribution:")
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"      {topic}: {count} comments")
            
            # Show detailed results
            print(f"\n   📋 Detailed results:")
            for i, result in enumerate(data.get('results', []), 1):
                print(f"      {i}. {result.get('body', '')[:50]}...")
                print(f"         → {result.get('topic', 'Unknown')} ({result.get('topic_score', 0):.3f})")
        else:
            print(f"❌ Full batch failed: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Full batch error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting DistilBART Classification Backend Tests")
    print(f"🎯 Target URL: {BACKEND_URL}")
    print("=" * 60)
    
    # Test sequence
    tests = [
        ("Root endpoint", test_root_endpoint),
        ("Health check", test_health_endpoint), 
        ("Small batch", test_small_batch),
        ("Full batch", test_full_batch)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    total_passed = sum(results.values())
    print(f"\n🎯 Overall: {total_passed}/{len(tests)} tests passed")
    
    if all(results.values()):
        print("🎉 All tests passed! Your backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()