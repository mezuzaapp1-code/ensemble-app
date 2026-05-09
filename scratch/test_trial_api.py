import requests
import uuid

BASE_URL = "http://localhost:8000"

def test_trial():
    # 1. Create session
    resp = requests.post(f"{BASE_URL}/session/new", json={"title": "Trial Test"})
    session_id = resp.json()["session_id"]
    print(f"Created session: {session_id}")
    
    # 2. Set profile
    requests.post(f"{BASE_URL}/session/{session_id}/profile", json={
        "user_name": "TestUser",
        "user_role": "tester"
    })
    print("Profile set to TestUser")
    
    # 3. Run ensemble 4 times
    for i in range(1, 6):
        print(f"Running question {i}...")
        resp = requests.post(f"{BASE_URL}/ensemble/run", json={
            "session_id": session_id,
            "question": f"Question {i}"
        })
        data = resp.json()
        if data.get("success"):
            print(f"Success! Trial count: {data.get('trial_count')}")
        else:
            print(f"Failed: {data.get('error')} (Trial exceeded: {data.get('trial_exceeded')})")
            break

if __name__ == "__main__":
    test_trial()
