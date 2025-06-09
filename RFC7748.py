"""
纯Python实现X25519算法，严格遵守RFC7748规范。

功能：
- 私钥clamping
- Montgomery Ladder标量乘法
- 32字节小端字节序转换
- 固定基点定义（9）
- 公钥生成
- 两方共享密钥计算
- RFC7748官方测试向量验证

无任何第三方依赖，只用Python内置功能。

参考：
- RFC7748 https://datatracker.ietf.org/doc/html/rfc7748
- Curve25519原始论文
"""

# Curve25519参数
_p = 2**255 - 19          # 有限域素数
_a24 = 121665             # (486662 - 2) / 4，用于montgomery梯度转换

def _decode_u_coordinate(u_bytes):
    """32字节小端转整数"""
    if len(u_bytes) != 32:
        raise ValueError("输入u坐标长度必须是32字节")
    return int.from_bytes(u_bytes, 'little') % _p

def _encode_u_coordinate(u_int):
    """整数转32字节小端"""
    if not (0 <= u_int < _p):
        raise ValueError("整数超出字段范围")
    return u_int.to_bytes(32, 'little')

def _clamp_scalar(scalar_bytes):
    """私钥Clamp，掩盖敏感位，符合RFC7748"""
    if len(scalar_bytes) != 32:
        raise ValueError("私钥长度必须是32字节")
    scalar = bytearray(scalar_bytes)
    scalar[0] &= 248          # 低3位清零
    scalar[31] &= 127         # 最高位清零
    scalar[31] |= 64          # 倒数第2位置1
    return bytes(scalar)

def _inv(x):
    """x的乘法逆元 mod p，利用费马小定理"""
    return pow(x, _p-2, _p)

def _x25519_scalar_mult(scalar_clamped, u):
    """
    核心Montgomery Ladder标量乘法
    scalar_clamped: 32字节私钥，已clamp
    u: 点坐标整数

    返回：标量乘法后u坐标整数
    """
    k = int.from_bytes(scalar_clamped, 'little')

    x1 = u
    x2 = 1
    z2 = 0
    x3 = u
    z3 = 1
    swap = 0

    for t in reversed(range(255)):
        k_t = (k >> t) & 1
        swap ^= k_t
        if swap:
            # 交换变量
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = k_t

        A = (x2 + z2) % _p
        AA = (A * A) % _p
        B = (x2 - z2) % _p
        BB = (B * B) % _p
        E = (AA - BB) % _p
        C = (x3 + z3) % _p
        D = (x3 - z3) % _p
        DA = (D * A) % _p
        CB = (C * B) % _p

        x3 = ((DA + CB) % _p)**2 % _p
        z3 = (x1 * ((DA - CB) % _p)**2) % _p
        x2 = (AA * BB) % _p
        z2 = (E * (AA + _a24 * E % _p)) % _p

    if swap:
        x2, x3 = x3, x2
        z2, z3 = z3, z2

    z2_inv = _inv(z2)
    return (x2 * z2_inv) % _p

def x25519(private_key_32bytes, peer_public_key_32bytes):
    """
    执行X25519算法，计算共享密钥
    private_key_32bytes: 32字节私钥
    peer_public_key_32bytes: 32字节对方公钥

    返回32字节共享密钥
    """
    scalar_clamped = _clamp_scalar(private_key_32bytes)
    u = _decode_u_coordinate(peer_public_key_32bytes)
    shared_u = _x25519_scalar_mult(scalar_clamped, u)
    return _encode_u_coordinate(shared_u)

def get_public_key(private_key_32bytes):
    """
    根据私钥生成对应公钥。基点固定为9。
    """
    base_point = (9).to_bytes(32, 'little')
    return x25519(private_key_32bytes, base_point)

# -- Test vectors from RFC7748 Section 6.1 --

def _test_vectors():
    # 示例1: 私钥、公钥、共享密钥（RFC7748测试向量）
    sk = bytes.fromhex("a546e36bf0527c9d3b16154b82465edd"
                       "62144c0ac1fc5a18506a2244ba449ac4")
    pk = bytes.fromhex("e6db6867583030db3594c1a424b15f7c"
                       "c76804a2f1f2ff38607e1162c16d21b8")
    expected_shared = bytes.fromhex("c3da55379de9c6908e94b3b027eedeae"
                                    "e0bc81e91f2f0f9e14cf9805d8838f60")

    computed_pk = get_public_key(sk)
    if computed_pk != pk:
        raise AssertionError("公钥计算不符RFC7748测试向量！")

    computed_shared = x25519(sk, pk)
    if computed_shared != expected_shared:
        raise AssertionError("共享密钥计算不符RFC7748测试向量！")

    print("RFC7748测试向量验证通过！")

if __name__ == "__main__":
    import os

    print("=== X25519示例运行 ===")
    # 随机生成两个私钥
    priv1 = os.urandom(32)
    priv2 = os.urandom(32)

    print("私钥1:", priv1.hex())
    print("私钥2:", priv2.hex())

    # 生成对应公钥
    pub1 = get_public_key(priv1)
    pub2 = get_public_key(priv2)

    print("公钥1:", pub1.hex())
    print("公钥2:", pub2.hex())

    # 双方生成共享密钥
    shared1 = x25519(priv1, pub2)
    shared2 = x25519(priv2, pub1)

    print("共享密钥1:", shared1.hex())
    print("共享密钥2:", shared2.hex())

    assert shared1 == shared2, "双方计算的共享密钥不一致！"
    print("双方共享密钥一致，Diffie-Hellman协商成功！\n")

    # 验证RFC7748官方测试向量
    _test_vectors()
