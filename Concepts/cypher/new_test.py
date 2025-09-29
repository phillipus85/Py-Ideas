#!/usr/bin/env python3
"""
Complete reproducible script to replicate the decryption pipeline used in the session.

What it does (exhaustively):
 - Loads the 20x6 ciphertext grid (you can change it).
 - Sets key = "GUINEVERE" by default (changeable).
 - Tries all 720 permutations of the 6 columns.
 - For each permutation tries:
     - Vigenere (stream over whole text)
     - Vigenere rowwise (restart key on each 6-letter row)
     - Vigenere column-major (read down columns left->right into a stream, decrypt, rebuild rows)
     - Beaufort (same 3 modes)
     - Autokey Vigenere rowwise and Beaufort autokey rowwise (optional)
 - Runs both "permute then decrypt" and "decrypt then permute" variants.
 - Scores outputs with a deterministic English-likeness heuristic (word hits + ngram cues + vowel ratio).
 - Outputs the top N unique candidate plaintexts (with method label, permutation used, score),
   prints them as 20 rows × 6 letters and saves JSON with full details.

This is the full pipeline used in the conversation; it is deterministic and prints/saves the same results.
"""

from itertools import permutations
import json
from math import log

# -----------------------------
# INPUTS: ciphertext grid and key
# -----------------------------
GRID_ROWS = [
    "VQKHSA", "CKMRNM", "EFSOAD", "DRCCJP", "NRETOA", "JSWVCH", "OFNEEO", "REYYNN", "GEBGRI", "RBKUST",
    "EYGNRS", "DLWQTE", "NIUVVO", "UNJMYO", "GTEBHM", "SFBWOR", "GQBPGH", "DIPJPJ", "AUYLHX", "CHHVGD"
]
KEY = "GUINEVERE"   # <- set the key used in the session
ROW_LEN = 6

# -----------------------------
# Alphabet helpers
# -----------------------------
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_idx = {c: i for i, c in enumerate(ALPHABET)}

# -----------------------------
# Cipher functions
# -----------------------------


def vigenere_decrypt_stream(ciphertext: str, key: str) -> str:
    key_idx = [char_to_idx[k] for k in key.upper()]
    out = []
    ki = 0
    for ch in ciphertext.upper():
        if ch in char_to_idx:
            cidx = char_to_idx[ch]
            pidx = (cidx - key_idx[ki % len(key_idx)]) % 26
            out.append(ALPHABET[pidx])
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)


def vigenere_decrypt_rowwise_rows(rows, key):
    out_rows = []
    key_idx = [char_to_idx[k] for k in key.upper()]
    for r in rows:
        ki = 0
        out = []
        for ch in r:
            if ch in char_to_idx:
                cidx = char_to_idx[ch]
                pidx = (cidx - key_idx[ki % len(key_idx)]) % 26
                out.append(ALPHABET[pidx])
                ki += 1
            else:
                out.append(ch)
        out_rows.append(''.join(out))
    return out_rows


def vigenere_decrypt_colstream_rows(rows, key):
    # read columns top->bottom left->right into stream, decrypt the stream,
    # then reconstruct rows by reading across the reconstructed columns.
    nrows = len(rows)
    ncols = len(rows[0])
    stream = ''.join(rows[r][c] for c in range(ncols) for r in range(nrows))
    dec = vigenere_decrypt_stream(stream, key)
    # split dec into columns of length nrows, then rebuild rows
    cols = [dec[i*nrows:(i+1)*nrows] for i in range(ncols)]
    rows_out = [''.join(cols[c][r] for c in range(ncols))
                for r in range(nrows)]
    return rows_out


def beaufort_decrypt_stream(ciphertext: str, key: str) -> str:
    key_idx = [char_to_idx[k] for k in key.upper()]
    out = []
    ki = 0
    for ch in ciphertext.upper():
        if ch in char_to_idx:
            cidx = char_to_idx[ch]
            pidx = (key_idx[ki % len(key_idx)] - cidx) % 26
            out.append(ALPHABET[pidx])
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)


def beaufort_decrypt_rowwise_rows(rows, key):
    out_rows = []
    key_idx = [char_to_idx[k] for k in key.upper()]
    for r in rows:
        ki = 0
        out = []
        for ch in r:
            if ch in char_to_idx:
                cidx = char_to_idx[ch]
                pidx = (key_idx[ki % len(key_idx)] - cidx) % 26
                out.append(ALPHABET[pidx])
                ki += 1
            else:
                out.append(ch)
        out_rows.append(''.join(out))
    return out_rows


def beaufort_decrypt_colstream_rows(rows, key):
    nrows = len(rows)
    ncols = len(rows[0])
    stream = ''.join(rows[r][c] for c in range(ncols) for r in range(nrows))
    dec = beaufort_decrypt_stream(stream, key)
    cols = [dec[i*nrows:(i+1)*nrows] for i in range(ncols)]
    rows_out = [''.join(cols[c][r] for c in range(ncols))
                for r in range(nrows)]
    return rows_out


def vigenere_autokey_rowwise(rows, key):
    out_rows = []
    for r in rows:
        ks = list(key.upper())
        out = []
        for ch in r:
            if ch in char_to_idx:
                k = ks.pop(0)
                pidx = (char_to_idx[ch] - char_to_idx[k]) % 26
                p = ALPHABET[pidx]
                out.append(p)
                ks.append(p)  # append plaintext char to keystream (autokey)
            else:
                out.append(ch)
        out_rows.append(''.join(out))
    return out_rows


def beaufort_autokey_rowwise(rows, key):
    out_rows = []
    for r in rows:
        ks = list(key.upper())
        out = []
        for ch in r:
            if ch in char_to_idx:
                k = ks.pop(0)
                pidx = (char_to_idx[k] - char_to_idx[ch]) % 26
                p = ALPHABET[pidx]
                out.append(p)
                ks.append(p)
            else:
                out.append(ch)
        out_rows.append(''.join(out))
    return out_rows


def permute_columns(rows, perm):
    return [''.join(r[i] for i in perm) for r in rows]


def rows_to_stream(rows):
    return ''.join(rows)


# -----------------------------
# Scoring heuristic (deterministic)
# -----------------------------
# small curated lists and ngram-like cues used in the session
COMMON_WORDS = ["THE", "AND", "CASTLE", "GUINEVERE", "FOREST", "DREAM", "HIDDEN", "KNIGHT",
                "FABLE", "SPEAK", "THERE", "BEYOND", "GATES", "KING", "SECRET", "DREAMS", "CASTLES", "HALLS"]
COMMON_BIGRAMS = ["TH", "HE", "IN", "ER", "AN", "RE",
                  "ED", "ON", "ES", "ST", "EN", "AT", "TE", "OR"]
COMMON_TRIGRAMS = ["THE", "AND", "ING", "ENT",
                   "ION", "HER", "FOR", "TER", "NTH", "INT"]


def score_text(text: str) -> float:
    t = text.upper()
    score = 0.0
    for w in COMMON_WORDS:
        score += t.count(w) * 25.0
    score += t.count("THE") * 5.0
    for bg in COMMON_BIGRAMS:
        score += t.count(bg) * 0.5
    for tg in COMMON_TRIGRAMS:
        score += t.count(tg) * 1.0
    vowels = sum(t.count(v) for v in "AEIOU")
    vr = vowels / max(1, len(t))
    score += max(0, (vr - 0.30)) * 10.0
    return score

# -----------------------------
# Main exhaustive search
# -----------------------------


def run_search(grid_rows, key, row_len=6, top_n=20):
    perms = list(permutations(range(row_len)))
    results = []  # list of (score, label, perm, stream, rows_list)

    for perm in perms:
        perm_rows = permute_columns(grid_rows, perm)
        # permute -> decrypt: Vigenere & Beaufort variants
        # VIG rowwise (key restarts each row)
        rows_vig_row = vigenere_decrypt_rowwise_rows(perm_rows, key)
        stream_vig_row = rows_to_stream(rows_vig_row)
        results.append((score_text(stream_vig_row),
                       "perm->VIG_rowwise", perm, stream_vig_row, rows_vig_row))

        # VIG stream continuous on row-major stream
        stream_vig = vigenere_decrypt_stream(rows_to_stream(perm_rows), key)
        rows_vig_stream = [stream_vig[i:i+row_len]
                           for i in range(0, len(stream_vig), row_len)]
        results.append((score_text(stream_vig), "perm->VIG_stream",
                       perm, stream_vig, rows_vig_stream))

        # VIG column-major (read columns top->bottom into stream, decrypt, reconstruct rows)
        rows_vig_col = vigenere_decrypt_colstream_rows(perm_rows, key)
        stream_vig_col = rows_to_stream(rows_vig_col)
        results.append((score_text(stream_vig_col),
                       "perm->VIG_colstream", perm, stream_vig_col, rows_vig_col))

        # Beaufort counterparts
        rows_beau_row = beaufort_decrypt_rowwise_rows(perm_rows, key)
        stream_beau_row = rows_to_stream(rows_beau_row)
        results.append((score_text(stream_beau_row),
                       "perm->BEAU_rowwise", perm, stream_beau_row, rows_beau_row))

        stream_beau = beaufort_decrypt_stream(rows_to_stream(perm_rows), key)
        rows_beau_stream = [stream_beau[i:i+row_len]
                            for i in range(0, len(stream_beau), row_len)]
        results.append((score_text(stream_beau), "perm->BEAU_stream",
                       perm, stream_beau, rows_beau_stream))

        rows_beau_col = beaufort_decrypt_colstream_rows(perm_rows, key)
        stream_beau_col = rows_to_stream(rows_beau_col)
        results.append((score_text(stream_beau_col),
                       "perm->BEAU_colstream", perm, stream_beau_col, rows_beau_col))

        # Autokey rowwise variants (sometimes helpful)
        rows_vig_ak = vigenere_autokey_rowwise(perm_rows, key)
        stream_vig_ak = rows_to_stream(rows_vig_ak)
        results.append((score_text(
            stream_vig_ak), "perm->VIG_autokey_rowwise", perm, stream_vig_ak, rows_vig_ak))

        rows_beau_ak = beaufort_autokey_rowwise(perm_rows, key)
        stream_beau_ak = rows_to_stream(rows_beau_ak)
        results.append((score_text(
            stream_beau_ak), "perm->BEAU_autokey_rowwise", perm, stream_beau_ak, rows_beau_ak))

    # decrypt -> permute: decrypt original grid first (all variants), then permute the decrypted rows
    dec_vig_row = vigenere_decrypt_rowwise_rows(grid_rows, key)
    dec_vig_col = vigenere_decrypt_colstream_rows(grid_rows, key)
    dec_beau_row = beaufort_decrypt_rowwise_rows(grid_rows, key)
    dec_beau_col = beaufort_decrypt_colstream_rows(grid_rows, key)
    dec_vig_ak = vigenere_autokey_rowwise(grid_rows, key)
    dec_beau_ak = beaufort_autokey_rowwise(grid_rows, key)

    for perm in perms:
        # VIG rowwise decrypted then permuted
        permuted = permute_columns(dec_vig_row, perm)
        results.append((score_text(rows_to_stream(permuted)),
                       "VIG->perm_rowwise", perm, rows_to_stream(permuted), permuted))

        permuted = permute_columns(dec_vig_col, perm)
        results.append((score_text(rows_to_stream(
            permuted)), "VIG->perm_colstream", perm, rows_to_stream(permuted), permuted))

        permuted = permute_columns(dec_beau_row, perm)
        results.append((score_text(rows_to_stream(
            permuted)), "BEAU->perm_rowwise", perm, rows_to_stream(permuted), permuted))

        permuted = permute_columns(dec_beau_col, perm)
        results.append((score_text(rows_to_stream(
            permuted)), "BEAU->perm_colstream", perm, rows_to_stream(permuted), permuted))

        # autokey decrypt-then-permute
        permuted = permute_columns(dec_vig_ak, perm)
        results.append((score_text(rows_to_stream(
            permuted)), "VIG_AK->perm_rowwise", perm, rows_to_stream(permuted), permuted))
        permuted = permute_columns(dec_beau_ak, perm)
        results.append((score_text(rows_to_stream(
            permuted)), "BEAU_AK->perm_rowwise", perm, rows_to_stream(permuted), permuted))

    # Rank results and keep unique streams
    results.sort(reverse=True, key=lambda x: x[0])
    unique = []
    seen = set()
    top = []
    for sc, label, perm, stream, rows in results:
        if stream in seen:
            continue
        seen.add(stream)
        top.append((sc, label, perm, stream, rows))
        if len(top) >= top_n:
            break

    return top


# -----------------------------
# Run search and produce files / printout
# -----------------------------
if __name__ == "__main__":
    top_candidates = run_search(GRID_ROWS, KEY, ROW_LEN, top_n=20)

    print(
        f"Found {len(top_candidates)} top unique candidates. Writing JSON and printing top 10.\n")
    # Save to JSON for inspection
    out = []
    for i, (sc, label, perm, stream, rows) in enumerate(top_candidates, start=1):
        out.append({"rank": i, "score": sc, "label": label,
                   "perm": perm, "rows": rows, "stream": stream})
    with open('top20_full_pipeline.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Print top 10 in a readable format (6-letter rows)
    for i, (sc, label, perm, stream, rows) in enumerate(top_candidates[:10], start=1):
        print(
            f"--- Candidate {i}: score={sc:.2f} label={label} perm={perm} ---")
        for r in rows:
            print(r)
        print()

    print("Saved full results to top20_full_pipeline.json")
