# Huffman Encoder & Decoder

A command-line utility for compressing and decompressing ASCII text files using Huffman coding. This tool encodes `.txt` files into a custom binary `.zhf` format and can decode them back with full accuracy.

## Features

- Huffman encoding and decoding
- Command-line interface
- Custom binary file format with versioning
- ASCII-only input validation
- Unit test support
- Ready for streaming input/output with large files

## Requirements

- Python 3.10+

## Usage

### Encode a file

```
python huffman.py encode --input input.txt --output output.zhf
```

### Decode a file

```
python huffman.py decode --input output.zhf --output decoded.txt
```

## File Format (.zhf)

The `.zhf` file contains:

- File format constant (4 bytes)
- File format version (1 byte)
- Frequency table length (4 bytes)
- Frequency table entries (1 byte char + 4 byte count each)
- Encoded message length (4 bytes)
- Final byte bit length (1 byte)
- Encoded data (variable)

## Testing

To run tests:

```
pytest
```

Make sure test files like `hello_world.txt` and `lorem.txt` are available or modify the tests to use temporary files.

## Limitations

- Only ASCII characters are supported
- This is for educational purposes, so I wouldn't use this in a real scenario
