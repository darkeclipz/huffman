# ZHF File Format Specification

The `.zhf` file format is a binary format used for storing compressed ASCII text using Huffman coding. It is designed for simplicity, efficient decoding, and support for streaming I/O.

## Overview

Each `.zhf` file consists of:

1. A file format identifier (magic number)
2. A version number
3. A frequency table used to reconstruct the Huffman tree
4. The size and structure of the encoded message

All integers are encoded in **big-endian** byte order.

## Layout

| Offset       | Size       | Description                          |
|--------------|------------|--------------------------------------|
| 0            | 4 bytes    | File format constant (`0x5A4846`)    |
| 4            | 1 byte     | File format version (`0x01`)         |
| 5            | 4 bytes    | Frequency table length `N`           |
| 9            | `N * 5`    | Frequency table entries              |
| 9 + N * 5    | 4 bytes    | Encoded message byte length `M`      |
| 13 + N * 5   | 1 byte     | Number of bits used in final byte    |
| 14 + N * 5   | `M` bytes  | Encoded message data                 |

## Details

### File Format Constant

A 4-byte magic number used to identify the file as a ZHF-compressed file.

```
0x5A4846  // ASCII "ZHF"
```

### File Format Version

A 1-byte version identifier. The current version is `1`.

### Frequency Table

- 4-byte integer: Number of entries `N` in the frequency table
- Each entry (5+L bytes):
  - 1 byte: character byte length `L`
  - L bytes: UTF-8 character bytes
  - 4 bytes: occurrence count

This table is used to reconstruct the exact Huffman tree used during compression.

### Encoded Message Info

- 4-byte integer: Total number of bytes used for the encoded message (`M`)
- 1 byte: Number of bits used in the final byte (1–8). The remaining bits in that byte should be ignored during decoding.

### Encoded Message Data

The actual Huffman-encoded message, packed bit-by-bit into `M` bytes.

## Example

Given an ASCII file with text:

```
hello
```

The frequency table might contain:

```
h: 1
e: 1
l: 2
o: 1
```

These would be stored in the frequency section as:

```
'h' (0x68), 0x00 00 00 01
'e' (0x65), 0x00 00 00 01
'l' (0x6C), 0x00 00 00 02
'o' (0x6F), 0x00 00 00 01
```


## Notes

- Only ASCII characters (values 0–127) are supported
- Decoders should validate the format constant and version before decoding
- Encoded bitstreams are read in MSB-first order (highest bit first per byte)
