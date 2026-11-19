"""Manifest hashing must not depend on the line endings git chose for the checkout."""
from src.core.bootstrap import content_hash, sha256_text_normalized


def test_crlf_and_lf_hash_identically(tmp_path):
    lf = tmp_path / "a.jsonl"
    crlf = tmp_path / "b.jsonl"
    lf.write_bytes(b'{"x": 1}\n{"y": 2}\n')
    crlf.write_bytes(b'{"x": 1}\r\n{"y": 2}\r\n')
    assert content_hash(lf) == content_hash(crlf)


def test_crlf_split_across_chunk_boundary(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_bytes(b"ab\r\ncd\r\n")
    # chunk=3 splits the first CRLF as "ab\r" | "\ncd..."
    assert sha256_text_normalized(p, chunk=3) == sha256_text_normalized(p)
    q = tmp_path / "d.jsonl"
    q.write_bytes(b"ab\ncd\n")
    assert sha256_text_normalized(p, chunk=3) == sha256_text_normalized(q)


def test_lone_cr_is_preserved(tmp_path):
    a = tmp_path / "e.txt"
    b = tmp_path / "f.txt"
    a.write_bytes(b"x\ry")
    b.write_bytes(b"xy")
    assert content_hash(a) != content_hash(b)


def test_binary_files_use_raw_bytes(tmp_path):
    a = tmp_path / "g.npy"
    b = tmp_path / "h.npy"
    a.write_bytes(b"\r\n")
    b.write_bytes(b"\n")
    assert content_hash(a) != content_hash(b)
