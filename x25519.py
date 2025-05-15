import os

# --------------------------------------------------------------
# 常量定义：Curve25519曲线固定参数
PRIME = 2**255 - 19      # 素数域，曲线定义基底
A24 = 121665             # 曲线参数 (A-2)/4，用于加速计算
BASE_POINT = 9           # Curve25519基点的X坐标，公钥计算默认用它

# --------------------------------------------------------------
def mod_inv(a, n=PRIME):
    """
    计算a在模n下的逆元素，即求 x 满足 (a * x) ≡ 1 (mod n)。
    采用费马小定理计算逆元：a^(n-2) mod n
    """
    return pow(a, n-2, n)

# --------------------------------------------------------------
def scalar_mult(k, u):
    """
    Curve25519核心算法：蒙哥马利曲线上的标量乘法
    
    输入:
        k - 标量，私钥
        u - 点的X坐标（基点或对方公钥）
    输出:
        k * u点的X坐标，也就是公钥或共享密钥计算结果
    实现细节：
        使用蒙哥马利梯度算法，其对防止侧信道攻击有优势。
    """
    x1 = u
    x2, z2 = 1, 0        # 初始化点(1, 0)对应曲线单位元
    x3, z3 = u, 1        # 初始化点(x1, 1)是原点u
    swap = 0             # 用于在算法步骤之间交换变量，防止泄露k

    for t in reversed(range(255)):   # 从第254位到第0位遍历k二进制
        k_t = (k >> t) & 1          # 取k的第t比特
        swap ^= k_t                 # 交换标志更新

        # 条件交换，根据swap决定是否交换x2,x3和z2,z3的值
        if swap:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = k_t

        # 曲线点加和与倍点运算
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
        z2 = (E * (AA + A24 * E % PRIME)) % PRIME

    # 最后一步，计算x2/z2 (通过模逆实现除法)
    z2_inv = mod_inv(z2, PRIME)
    return (x2 * z2_inv) % PRIME

# --------------------------------------------------------------
def generate_private_key():
    """
    生成Curve25519私钥，包含必要的掩码处理，
    保证私钥满足安全要求（固定高低位）。
    
    返回：
        0 <= k < 2^255，且满足一定的bit条件
    """
    random_bytes = os.urandom(32)   # 随机32字节
    k = int.from_bytes(random_bytes, 'little')  # 转成整数(小端)
    
    # Bit掩码操作，确保私钥约束
    # 清除最低3位，使得私钥是8的倍数
    k &= ~(0b111)
    # 设置第254位为1（确保私钥不小于2^254）
    k |= (1 << 254)
    # 清除最高位（第255位），保证私钥无限大
    k &= ~(1 << 255)
    
    return k

# --------------------------------------------------------------
def generate_public_key(private_key):
    """
    根据私钥计算公钥（标量乘基点）
    
    参数：
        private_key - Curve25519私钥
    返回：
        对应公钥的整数表示（点的X坐标）
    """
    return scalar_mult(private_key, BASE_POINT)

# --------------------------------------------------------------
def generate_shared_secret(private_key, peer_public_key):
    """
    在密钥交换过程中，利用自己的私钥与对方公钥生成共享密钥。
    
    参数:
        private_key - 本方私钥
        peer_public_key - 对方公钥
    返回:
        共享密钥（整数）
    """
    return scalar_mult(private_key, peer_public_key)

# --------------------------------------------------------------
if __name__ == "__main__":
    print("======= Curve25519密钥交换示例 =======")

    # 1. Alice生成自己的私钥和公钥
    alice_priv = generate_private_key()
    alice_pub = generate_public_key(alice_priv)
    print(f"Alice私钥(整数)：{alice_priv}")
    print(f"Alice公钥(整数)：{alice_pub}")
    print()

    # 2. Bob生成自己的私钥和公钥
    bob_priv = generate_private_key()
    bob_pub = generate_public_key(bob_priv)
    print(f"Bob私钥(整数)：{bob_priv}")
    print(f"Bob公钥(整数)：{bob_pub}")
    print()

    # 3. Alice使用自己的私钥和Bob的公钥计算共享密钥
    alice_shared = generate_shared_secret(alice_priv, bob_pub)
    print(f"Alice计算的共享密钥：{alice_shared}")

    # 4. Bob使用自己的私钥和Alice的公钥计算共享密钥
    bob_shared = generate_shared_secret(bob_priv, alice_pub)
    print(f"Bob计算的共享密钥：{bob_shared}")

    # 5. 验证共享密钥是否一致
    if alice_shared == bob_shared:
        print("成功！双方共享密钥一致。")
    else:
        print("失败！共享密钥不一致。")



"""
The public key is generated from the private key through scalar multiplication on the elliptic curve. 
Specifically, a fixed base point is used for this scalar multiplication (for X25519, this base point has a coordinate of 9). 
This means that the public key is obtained by multiplying the base point by the private key scalar.
Alice computes the shared secret using Bob's public key
SharedSecret_A =  A_sk * B_pk
Bob computes the shared secret using Alice's public key
SharedSecret_B =  B_sk * A_pk
Bob computes the shared secret using Alice's public key:
SharedSecret_B = B_sk * A_pk
Due to the properties of elliptic curve multiplication, SharedSecret_A and SharedSecret_B will be equal. 
This property allows both parties to establish a common secret even over an insecure channel.
"""
print("Shared secret matches:", alice_shared_secret)
