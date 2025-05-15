import os
import hashlib

# Curve25519素数域和参数
PRIME = 2 ** 255 - 19  # 素数域 p = 2^255 - 19
A24 = 121665           # Curve参数 (486662 - 2)//4，用于优化计算
BASE_POINT = 9         # Curve25519基点的X坐标，固定为9

# 将32字节小端字节串转换成整数
def bytes_to_int_le(b):
    return int.from_bytes(b, "little")

# 将整数转换为32字节小端字节串
def int_to_bytes_le(i):
    return i.to_bytes(32, "little")

# 私钥clamping过程，保证私钥符合Curve25519安全规范
def clamp_scalar(k_bytes):
    k = bytearray(k_bytes)   # 转为可变字节数组
    k[0] &= 248              # 清空最低3位（bit 0,1,2清零）
    k[31] &= 127             # 清最高位（bit 255清零）
    k[31] |= 64              # 置第254位为1
    return bytes(k)          # 返回不可变bytes对象

# 计算模逆元，使用费马小定理
def mod_inv(a, p=PRIME):
    return pow(a, p - 2, p)

# Curve25519蒙哥马利标量乘法核心，实现 k * u （k为标量，u为点X坐标）
def scalar_mult(k, u):
    # 初始化变量，代表坐标点
    x1 = u                       # 输入点的X坐标
    x2, z2 = 1, 0                # 算法中代表点“单位元”，正式项目对应曲线加法单位元
    x3, z3 = u, 1                # 当前处理点的坐标，初始为输入点
    swap = 0                     # 交换指示位，防止侧信道攻击通过条件分支泄露信息

    # 迭代标量k的255位（大端，从高位到低位）
    for t in reversed(range(255)):
        k_t = (k >> t) & 1       # 取标量k的第t位
        swap ^= k_t              # 与swap异或，决定是否交换变量值以隐蔽算法操作
        if swap:
            # 条件交换x2 <-> x3, z2 <-> z3
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = k_t               # 更新swap为当前k位，用于下一循环使用

        # Curve25519蒙哥马利梯度算法组分
        A = (x2 + z2) % PRIME        # A = x2 + z2
        AA = (A * A) % PRIME         # AA = A^2
        B = (x2 - z2) % PRIME        # B = x2 - z2
        BB = (B * B) % PRIME         # BB = B^2
        E = (AA - BB) % PRIME        # E = AA - BB
        C = (x3 + z3) % PRIME        # C = x3 + z3
        D = (x3 - z3) % PRIME        # D = x3 - z3
        DA = (D * A) % PRIME         # DA = D * A
        CB = (C * B) % PRIME         # CB = C * B

        # 计算新x3和z3坐标
        x3 = ((DA + CB) ** 2) % PRIME   # x3 = (DA + CB)^2 mod p
        z3 = ((DA - CB) ** 2) % PRIME   # z3 = (DA - CB)^2 mod p
        x3 = (x3 * x1) % PRIME           # x3 = x3 * x1 mod p

        # 计算新x2和z2坐标
        x2 = (AA * BB) % PRIME           # x2 = AA * BB mod p
        z2 = (E * ((AA + A24 * E) % PRIME)) % PRIME  # z2 = E * (AA + A24*E) mod p

    # 循环结束后，返回x2 / z2作为标量乘结果点X坐标
    z2_inv = mod_inv(z2)                 # 计算z2的模逆
    result = (x2 * z2_inv) % PRIME      # x2 * z2_inv mod p，即x2/z2模p
    return result

# 下面代码为演示流程，按顺序执行

# 1. 生成Alice的私钥（32字节随机数，后经过clamping）
raw_priv_alice = os.urandom(32)
priv_alice = clamp_scalar(raw_priv_alice)

# 2. 将私钥转成整数
priv_alice_int = bytes_to_int_le(priv_alice)

# 3. 计算Alice的公钥 = alice_priv * base_point
pub_alice_int = scalar_mult(priv_alice_int, BASE_POINT)

# 4. 公钥转32字节小端bytes
pub_alice = int_to_bytes_le(pub_alice_int)

print("Alice私钥:", priv_alice.hex())
print("Alice公钥:", pub_alice.hex())

# 5. 生成Bob的私钥和公钥，同上
raw_priv_bob = os.urandom(32)
priv_bob = clamp_scalar(raw_priv_bob)
priv_bob_int = bytes_to_int_le(priv_bob)
pub_bob_int = scalar_mult(priv_bob_int, BASE_POINT)
pub_bob = int_to_bytes_le(pub_bob_int)

print("Bob私钥:", priv_bob.hex())
print("Bob公钥:", pub_bob.hex())

# 6. Alice使用自己的私钥和Bob公钥计算共享密钥
pub_bob_int_for_alice = bytes_to_int_le(pub_bob)
shared_alice_int = scalar_mult(priv_alice_int, pub_bob_int_for_alice)
shared_alice_bytes = int_to_bytes_le(shared_alice_int)

# 7. Bob使用自己的私钥和Alice公钥计算共享密钥
pub_alice_int_for_bob = bytes_to_int_le(pub_alice)
shared_bob_int = scalar_mult(priv_bob_int, pub_alice_int_for_bob)
shared_bob_bytes = int_to_bytes_le(shared_bob_int)

# 8. 对共享密钥使用SHA-256哈希，产生最终对称密钥
sym_key_alice = hashlib.sha256(shared_alice_bytes).digest()
sym_key_bob = hashlib.sha256(shared_bob_bytes).digest()

print("Alice共享密钥(SHA256):", sym_key_alice.hex())
print("Bob共享密钥(SHA256):  ", sym_key_bob.hex())

# 9. 判断是否一致
if sym_key_alice == sym_key_bob:
    print("密钥交换成功，双方共享密钥一致")
else:
    print("错误，共享密钥不一致！")
