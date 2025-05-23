import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

# 1. DH 密钥交换
def generate_dh_keypair():
    """生成 X25519 密钥对，用于 Diffie-Hellman 密钥交换."""
    private_key = x25519.X25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key

def exchange_dh_keys(private_key, other_public_key):
    """使用 Diffie-Hellman 密钥交换算法，计算共享密钥。"""
    shared_key = private_key.exchange(other_public_key)
    return shared_key

# 2. 密钥派生 (HKDF)
def derive_key(shared_key, salt=None, info=b'handshake data', key_length=32):
    """使用 HKDF 从共享秘密派生密钥。"""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        info=info
    )
    return hkdf.derive(shared_key)

# 3. 加密和解密 (Fernet - AES 加密)
def encrypt_message(message, key):
    """使用 Fernet (基于 AES 的认证加密) 加密消息。"""
    f = Fernet(key)
    encrypted_message = f.encrypt(message.encode())
    return encrypted_message

def decrypt_message(encrypted_message, key):
    """使用 Fernet 解密消息。"""
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message).decode()
    return decrypted_message

# 模拟 Double Ratchet 中的 KDF 链
def kdf_ratchet(chain_key):
    """模拟 Double Ratchet 算法中的 KDF 链。"""
    message_key = derive_key(chain_key, info=b"message_key")
    new_chain_key = derive_key(chain_key, info=b"chain_key")
    return new_chain_key, message_key

# 以下函数用于演示端到端加密流程
def simulate_e2ee_session(message="Hello, Bob! This is a secret message."):
    """模拟 Alice 和 Bob 之间的端到端加密会话。"""

    # Alice 和 Bob 生成 DH 密钥对
    alice_private_key, alice_public_key = generate_dh_keypair()
    bob_private_key, bob_public_key = generate_dh_keypair()

    # Alice 和 Bob 交换公钥 (假设安全地交换)
    # 在实际应用中，需要进行身份验证，以防止中间人攻击

    # Alice 和 Bob 计算 DH 共享密钥
    alice_shared_key = exchange_dh_keys(alice_private_key, bob_public_key)
    bob_shared_key = exchange_dh_keys(bob_private_key, alice_public_key)

    # 派生初始根密钥 (Root Key)
    alice_root_key = derive_key(alice_shared_key, salt=os.urandom(16), info=b"root_key")
    bob_root_key = derive_key(bob_shared_key, salt=os.urandom(16), info=b"root_key")

    # 从根密钥派生初始链密钥
    alice_chain_key, _ = kdf_ratchet(alice_root_key)
    bob_chain_key, _ = kdf_ratchet(bob_root_key)

    # Alice 发送消息
    alice_chain_key, message_key = kdf_ratchet(alice_chain_key)  # 密钥轮换
    encrypted_message = encrypt_message(message, message_key)  # 使用新的消息密钥加密

    # Bob 接收消息
    bob_chain_key, message_key = kdf_ratchet(bob_chain_key)  # 密钥轮换
    decrypted_message = decrypt_message(encrypted_message, message_key)  # 使用新的消息密钥解密

    return encrypted_message, decrypted_message

# 使用示例：
if __name__ == '__main__':
    encrypted, decrypted = simulate_e2ee_session()

    print("Encrypted Message:", encrypted)
    print("Decrypted Message:", decrypted)
