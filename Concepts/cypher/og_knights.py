import re
import itertools
from string import ascii_uppercase

# -----------------------------
# Utility functions
# -----------------------------


def vigenere_decrypt(cipher, key):
    plain = []
    klen = len(key)
    for i, c in enumerate(cipher):
        if c not in ascii_uppercase:  # skip non-letters
            plain.append(c)
            continue
        shift = ord(key[i % klen]) - ord('A')
        p = (ord(c) - ord('A') - shift) % 26
        plain.append(chr(p + ord('A')))
    return "".join(plain)


def beaufort_decrypt(cipher, key):
    plain = []
    klen = len(key)
    for i, c in enumerate(cipher):
        if c not in ascii_uppercase:
            plain.append(c)
            continue
        shift = ord(key[i % klen]) - ord('A')
        p = (shift - (ord(c) - ord('A'))) % 26
        plain.append(chr(p + ord('A')))
    return "".join(plain)


def transpose_rows(text, row_len):
    return ''.join(text[i::row_len] for i in range(row_len))


def transpose_cols(text, row_len):
    rows = [text[i:i+row_len] for i in range(0, len(text), row_len)]
    return ''.join(''.join(r[i] for r in rows if i < len(r)) for i in range(row_len))


# scoring function (very naive: count dictionary words)


def score_text(text, wordlist):
    words = re.findall(r'[A-Z]+', text)
    hits = sum(1 for w in words if w.lower() in wordlist)
    return hits

# -----------------------------
# Main candidate generator
# -----------------------------


def generate_candidates(cipher, key, row_len, wordlist, topn=20):
    candidates = []

    # Step 1: define transformations
    methods = {
        "Vigenere": vigenere_decrypt,
        "Beaufort": beaufort_decrypt,
    }

    transpositions = {
        "None": lambda t: t,
        "RowWise": lambda t: transpose_rows(t, row_len),
        "ColWise": lambda t: transpose_cols(t, row_len),
    }

    positions = {
        "KeyBefore": lambda f, t, k: f(transpositions[t](cipher), k),
        "KeyAfter": lambda f, t, k: transpositions[t](f(cipher, k)),
    }

    # Step 2: try all permutations
    for method_name, func in methods.items():
        for trans_name, trans_func in transpositions.items():
            for pos_name, pos_func in positions.items():
                candidate = pos_func(func, trans_name, key)
                s = score_text(candidate, wordlist)
                candidates.append(
                    (s, f"{method_name}-{trans_name}-{pos_name}", candidate))

    # Step 3: rank and return top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:topn]


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example ciphertext (replace with yours)
    ciphertext = """
VQKHSA
CKMRNM
EFSOAD
DRCCJP
NRETOA
JSWVCH
OFNEEO
REYYNN
GEBGRI
RBKUST
EYGNRS
DLWQTE
NIUVVO
UNJMYO
GTEBHM
SFBWOR
GQBPGH
DIPJPJ
AUYLHX
CHHVGD
    """  # put your full text here
    key = "SECRET"  # example key

    # Load a small English wordlist (use your own larger list for better scoring)
    wordlist = {"there", "castle", "secret", "dreams",
                "fables", "land", "ancient", "speak", "kings"}

    results = generate_candidates(ciphertext.replace(
        " ", ""), key, row_len=6, wordlist=wordlist)

    for i, (score, label, text) in enumerate(results, 1):
        print(f"{i:2d}. [{score}] {label}\n{text}\n")
