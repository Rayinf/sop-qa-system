#!/usr/bin/env python3
import requests
import json

def test_login_and_kb_api():
    base_url = "http://localhost:8000"
    
    # 1. 测试登录
    print("1. 测试用户登录...")
    login_data = {
        "username": "admin@example.com",
        "password": "admin123456"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        print(f"登录响应: {response.status_code}")
        
        if response.status_code == 200:
            login_result = response.json()
            access_token = login_result.get("access_token")
            print(f"登录成功，获取到 token: {access_token[:20]}...")
            
            # 2. 使用 token 测试知识库 API
            print("\n2. 测试知识库 API...")
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            # 测试获取知识库列表
            kb_response = requests.get(f"{base_url}/api/v1/kb", headers=headers)
            print(f"知识库 API 响应: {kb_response.status_code}")
            
            if kb_response.status_code == 200:
                kb_data = kb_response.json()
                print(f"知识库数据: {json.dumps(kb_data, indent=2, ensure_ascii=False)}")
                print("✅ 知识库 API 测试成功！")
            else:
                print(f"❌ 知识库 API 失败: {kb_response.text}")
                
        else:
            print(f"❌ 登录失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    test_login_and_kb_api()