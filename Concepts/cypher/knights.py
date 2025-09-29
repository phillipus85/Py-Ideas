import re
import os
# import itertools
from string import ascii_uppercase

# -----------------------------
# Utility functions
# -----------------------------


def vigenere_decrypt(cipher: str, key: str) -> str:
    plain = []
    klen = len(key)
    for i, c in enumerate(cipher):
        # skip non-letters
        if c not in ascii_uppercase:
            plain.append(c)
            continue
        # else:
        shift = ord(key[i % klen]) - ord("A")
        p = (ord(c) - ord("A") - shift) % 26
        plain.append(chr(p + ord("A")))
    return "".join(plain)


def beaufort_decrypt(cipher: str, key: str) -> str:
    plain = []
    klen = len(key)
    for i, c in enumerate(cipher):
        if c not in ascii_uppercase:
            plain.append(c)
            continue
        shift = ord(key[i % klen]) - ord("A")
        p = (shift - (ord(c) - ord("A"))) % 26
        plain.append(chr(p + ord("A")))
    return "".join(plain)


def transpose_rows(txt: str, row_len: int) -> str:
    # list comprehension version
    # return "".join(txt[i::row_len] for i in range(row_len))

    # expanded version
    ans = []
    for i in range(row_len):
        ans.append(txt[i::row_len])
    return "".join(ans)


def transpose_cols(txt: str, row_len: int) -> str:
    # list comprehension version
    # rows = [txt[i:i + row_len] for i in range(0, len(txt), row_len)]
    # return "".join("".join(r[i] for r in rows if i < len(r)) for i in range(row_len))

    # expanded version
    rows = []
    for i in range(0, len(txt), row_len):
        rows.append(txt[i:i + row_len])
    ans = []
    for i in range(row_len):
        for r in rows:
            if i < len(r):
                ans.append(r[i])
    return "".join(ans)


# scoring function (very naive: count dictionary words)


def score_text(txt: str, word_lt: set) -> int:
    words = re.findall(r"[A-Z]+", txt)
    score = [1 for w in words if w.lower() in word_lt]
    hits = sum(score)
    return hits

# -----------------------------
# candidate generator
# -----------------------------


def generate_candidates(cipher: str,
                        key: str,
                        row_len: int,
                        word_lt: set,
                        top: int = 20) -> list:
    candidates = []

    # Step 1: define transformations
    methods = {
        "Vigenere": vigenere_decrypt,
        "Beaufort": beaufort_decrypt,
    }

    transpositions = {
        "None": lambda t: t,
        "row_wise": lambda t: transpose_rows(t, row_len),
        "col_wise": lambda t: transpose_cols(t, row_len),
    }

    positions = {
        "key_before": lambda f, t, k: f(transpositions[t](cipher), k),
        "key_after": lambda f, t, k: transpositions[t](f(cipher, k)),
    }

    # Step 2: try all permutations
    for _mkey, _method in methods.items():
        for _tkey, _trans in transpositions.items():
            for _pkey, _pos in positions.items():
                # Apply the position strategy with the correct method and transposition function
                candidate = _pos(_method, _trans, key)
                s = score_text(candidate, word_lt)
                _eval = (s, f"{_mkey}-{_tkey}-{_pkey}", candidate)
                candidates.append(_eval)

    # Step 3: rank and return top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:top]


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # get current python folder path
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # load cypher text
    fp = os.path.join(cur_dir, "msg.txt")
    cypher_txt = open(fp, "r", encoding="utf-8")
    print(f"Cypher Text:\n{cypher_txt.read()}\n")

    # secret key
    key = "GUINEVERE"  # example key

    # Load a small word list for scoring
    word_lt = set()
    fp = os.path.join(cur_dir, "words.txt")
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            word_lt.add(line.strip().lower())

    print(f"Loaded {len(word_lt)} words for scoring.\n")
    print(f"Sample words: {list(word_lt)[:10]}\n")

    # Generate candidates
    prep = cypher_txt.read().replace(" ", "")
    results = generate_candidates(prep,
                                  key,
                                  row_len=6,
                                  word_lt=word_lt)

    for i, (score, label, txt) in enumerate(results, 1):
        print(f"{i:2d}. [{score}] {label}\n{txt}\n")
