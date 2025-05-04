import os

# Constants specific to Curve25519
PRIME = 2**255 - 19
A24 = 121665
# --------------------------------------------------------------
# Function to perform modular inversion
def mod_inv(a, n=PRIME):
    return pow(a, n - 2, n)
# --------------------------------------------------------------
# Scalar multiplication on Curve25519
def scalar_mult(k, u):
    x_1 = u
    x_2, z_2 = 1, 0
    x_3, z_3 = u, 1
    swap = 0
    t_range = range(255)
    for t in reversed(t_range):
        k_t = (k >> t) & 1
        swap ^= k_t
        
        x_2, x_3 = (x_3, x_2) if swap else (x_2, x_3)
        z_2, z_3 = (z_3, z_2) if swap else (z_2, z_3)
        
        swap = k_t
        
        A = (x_2 + z_2) % PRIME
        AA = (A * A) % PRIME
        B = (x_2 - z_2) % PRIME
        BB = (B * B) % PRIME
        E = (AA - BB) % PRIME
        C = (x_3 + z_3) % PRIME
        D = (x_3 - z_3) % PRIME
        DA = (D * A) % PRIME
        CB = (C * B) % PRIME
        
        x_3 = (DA + CB) % PRIME
        x_3 = (x_3 * x_3) % PRIME
        z_3 = (DA - CB) % PRIME
        z_3 = (z_3 * z_3) % PRIME
        x_3 = (x_3 * x_1) % PRIME

        x_2 = (AA * BB) % PRIME
        z_2 = (E * ((AA + A24 * E) % PRIME)) % PRIME

    z_2_inv = mod_inv(z_2)
    return (x_2 * z_2_inv) % PRIME

# --------------------------------------------------------------
# Function to generate a private key
def generate_private_key():
    private_key = os.urandom(32)
    k = int.from_bytes(private_key, 'little')
    k &= ~0b111
    k |= 0b1000000000000000000000000000000000000000000000000000000000000000
    return k
# --------------------------------------------------------------
# Function to compute a public key from a private key
def generate_public_key(private_key):
    base_point = 9
    return scalar_mult(private_key, base_point)
# --------------------------------------------------------------
# Function to generate a shared secret
def generate_shared_secret(private_key, peer_public_key):
    return scalar_mult(private_key, peer_public_key)
# --------------------------------------------------------------
# Example process for Alice and Bob
alice_private = generate_private_key()
alice_public = generate_public_key(alice_private)
bob_private = generate_private_key()
bob_public = generate_public_key(bob_private)
# --------------------------------------------------------------
# Exchange public keys (alice_public, bob_public)
alice_shared_secret = generate_shared_secret(alice_private, bob_public)
bob_shared_secret = generate_shared_secret(bob_private, alice_public)
# --------------------------------------------------------------
# Both parties should derive the same shared secret
assert alice_shared_secret == bob_shared_secret
print("Shared secret matches:", alice_shared_secret)
