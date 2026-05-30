import hashlib
import json
import os

SALT = "Lease_Project_2026"
def hash_password(password, salt=SALT):
    """加盐哈希计算"""
    return hashlib.sha256((password + salt).encode()).hexdigest()


# 初始账号配置
initial_users = {
    "admin": {
        "password_hash": hash_password("Manager2026"),
        "role": "manager",
        "name": "财务经理"
    },
    "user": {
        "password_hash": hash_password("Staff2026"),
        "role": "viewer",
        "name": "普通会计"
    }
}

with open("users.json", "w", encoding="utf-8") as f:
    json.dump(initial_users, f, indent=4, ensure_ascii=False)

USERS_FILE = "users.json"

def add_user(username, password, role, name):
    # 加载现有用户
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
    else:
        users = {}

    # 添加新用户
    users[username] = {
        "password_hash": hash_password(password),
        "role": role, # 'manager' 或 'viewer'
        "name": name
    }

    # 写回文件
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)
    print(f"✅ 用户 [{username}] 已成功添加/更新。")
