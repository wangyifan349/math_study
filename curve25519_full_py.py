import os
import hashlib

PRIME = 2 ** 255 - 19
A24 = 121665
BASE_POINT = 9

def clamp_scalar(scalar_bytes):
    """
    Curve25519私钥clamping标准步骤：
    1) 清除最低三位（bit 0,1,2）
    2) 把倒数第2位 (bit 254) 置1
    3) 把最高位 (bit 255) 置0
    Args:
        scalar_bytes: 32字节私钥 (bytes)
    Returns:
        clamped_bytes: 32字节clamped私钥 (bytes)
    """
    scalar = bytearray(scalar_bytes)
    scalar[0] &= 248      # 清除bit 0,1,2：保持最低5位固定
    scalar[31] &= 127     # 清除bit 255最高位
    scalar[31] |= 64      # 设置bit 254
    return bytes(scalar)

def bytes_to_int_le(b):
    """小端字节转整数"""
    return int.from_bytes(b, "little")

def int_to_bytes_le(i):
    """整数转32字节小端bytes"""
    return i.to_bytes(32, "little")

def mod_inv(a, p=PRIME):
    """模逆"""
    return pow(a, p - 2, p)

def scalar_mult(k_int, u_int):
    """Curve25519蒙哥马利标量乘法，输入整数，输出整数"""
    x1 = u_int
    x2, z2 = 1, 0
    x3, z3 = u_int, 1
    swap = 0
    for t in reversed(range(255)):
        k_t = (k_int >> t) & 1
        swap ^= k_t
        if swap:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = k_t
        A = (x2 + z2) % PRIME
        AA = (A * A) % PRIME
        B = (x2 - z2) % PRIME
        BB = (B * B) % PRIME
        E = (AA - BB) % PRIME
        C = (x3 + z3) % PRIME
        D = (x3 - z3) % PRIME
        DA = (D * A) % PRIME
        CB = (C * B) % PRIME

        x3 = ((DA + CB) ** 2) % PRIME
        z3 = ((DA - CB) ** 2) % PRIME
        x3 = (x3 * x1) % PRIME

        x2 = (AA * BB) % PRIME
        z2 = (E * ((AA + A24 * E) % PRIME)) % PRIME
    
    z2_inv = mod_inv(z2)
    return (x2 * z2_inv) % PRIME

def generate_private_key():
    """生成32字节私钥并clamping，返回bytes"""
    raw = os.urandom(32)
    return clamp_scalar(raw)

def generate_public_key(private_bytes):
    """传入私钥bytes，返回公钥bytes"""
    priv_int = bytes_to_int_le(private_bytes)
    pub_int = scalar_mult(priv_int, BASE_POINT)
    return int_to_bytes_le(pub_int)

def generate_shared_secret(private_bytes, peer_public_bytes):
    """
    使用自己的私钥和对方公钥生成共享密钥
    输入和输出均为bytes
    """
    priv_int = bytes_to_int_le(private_bytes)
    pub_int = bytes_to_int_le(peer_public_bytes)
    shared_int = scalar_mult(priv_int, pub_int)
    shared_bytes = int_to_bytes_le(shared_int)
    # 简单对共享密钥做SHA256作为KDF，输出32字节长度密钥
    return hashlib.sha256(shared_bytes).digest()

if __name__ == "__main__":
    print("=== Curve25519密钥交换（纯Python实现）===")

    # Alice生成密钥对
    alice_priv = generate_private_key()
    alice_pub = generate_public_key(alice_priv)
    print(f"Alice私钥(HEX): {alice_priv.hex()}")
    print(f"Alice公钥(HEX): {alice_pub.hex()}")

    # Bob生成密钥对
    bob_priv = generate_private_key()
    bob_pub = generate_public_key(bob_priv)
    print(f"Bob私钥(HEX): {bob_priv.hex()}")
    print(f"Bob公钥(HEX): {bob_pub.hex()}")

    # 计算共享密钥
    alice_shared = generate_shared_secret(alice_priv, bob_pub)
    bob_shared = generate_shared_secret(bob_priv, alice_pub)
    print(f"Alice共享密钥(HEX): {alice_shared.hex()}")
    print(f"Bob共享密钥(HEX):   {bob_shared.hex()}")

    if alice_shared == bob_shared:
        print("成功！双方共享密钥一致！")
    else:
        print("失败！共享密钥不一致！")
