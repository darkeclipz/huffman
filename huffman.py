from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Generator, List
import io
import pathlib
from collections import Counter
import argparse
from tqdm import tqdm
import heapq
from bitwriter import BufferedBitWriterNative

FILE_FORMAT_CONSTANT = 0x5A4846
FILE_FORMAT_VERSION = 2

@dataclass
class Node:
    id: int
    parent: Node | None
    char: str | None
    weight: int
    children: List[Node]
    encoding_path: tuple[int, int] | None = None
    _id_counter: ClassVar[int] = 0
    @staticmethod
    def new_leaf(character: str | None, weight: int):
        Node._id_counter += 1
        return Node(Node._id_counter, None, character, weight, [])
    @staticmethod
    def new_node(weight: int, left: Node, right: Node):
        Node._id_counter += 1
        return Node(Node._id_counter, None, None, weight, [left, right])
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    def __lt__(self, other):
        return (self.weight, self.id) < (other.weight, other.id)

def write_int8(stream: io.BufferedWriter, value: int) -> None:
    stream.write(value.to_bytes(1, "big"))

def write_int32(stream: io.BufferedWriter, value: int) -> None:
    stream.write(value.to_bytes(4, "big"))

def read_int8(stream: io.BufferedReader) -> int:
    return int.from_bytes(stream.read(1), "big")

def read_int32(stream: io.BufferedReader) -> int:
    return int.from_bytes(stream.read(4), "big")

class HuffmanTree:
    def __init__(self, root: Node, encoding_table: dict[str, Node]):
        self.root = root
        self.encoding_table = encoding_table
        self.compute_bit_sequences()
    def compute_bit_sequences(self):
        for _, node in self.encoding_table.items():
            current = node
            bits = []
            while current.parent:
                parent = current.parent
                bit = parent.children.index(current)
                bits.append(bit)
                current = parent
            bits.reverse()
            path_int = 0
            for b in bits:
                path_int = (path_int << 1) | b
            node.encoding_path = (path_int, len(bits))
    @staticmethod
    def new(frequency_list: List[tuple[str, int]]):
        encoding_table: dict[str, Node] = {}
        heap: List[Node] = []
        for character, frequency in frequency_list:
            node = Node.new_leaf(character, frequency)
            heapq.heappush(heap, node)
            encoding_table[character] = node
        while len(heap) >= 2:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged_node = Node.new_node(left.weight + right.weight, left=left, right=right)
            left.parent = merged_node
            right.parent = merged_node
            heapq.heappush(heap, merged_node)
        return HuffmanTree(heap[0], encoding_table)

class HuffmanEncoderUnicode:
    def __init__(self, tree: HuffmanTree, stream: io.BufferedWriter):
        self.tree = tree
        self.stream = stream
        self.write_header()
        self.buffer = BufferedBitWriterNative(stream)
    def write_header(self):
        write_int32(self.stream, FILE_FORMAT_CONSTANT)
        write_int8(self.stream, FILE_FORMAT_VERSION)
        encoding_table_size = len(self.tree.encoding_table)
        write_int32(self.stream, encoding_table_size)
        for char, node in self.tree.encoding_table.items():
            encoded_char = char.encode("utf-8")
            write_int8(self.stream, len(encoded_char))
            self.stream.write(encoded_char)
            write_int32(self.stream, node.weight)
        self.encoded_message_size_position = self.stream.tell()
        write_int32(self.stream, 0)
        self.encoded_message_used_bits_position = self.stream.tell()
        write_int8(self.stream, 0)
    def write_chunk(self, text_chunk):
        self.buffer.write_chunk(text_chunk, self.tree.encoding_table)
    def close(self):
        last_pos = self.stream.tell()
        used_bits = self.buffer.bit_pos
        self.buffer.flush()
        total_bytes_written = self.stream.tell() - self.encoded_message_used_bits_position - 1
        self.stream.seek(self.encoded_message_size_position)
        write_int32(self.stream, total_bytes_written)
        self.stream.seek(self.encoded_message_used_bits_position)
        write_int8(self.stream, used_bits)
        self.stream.seek(last_pos)

class BitReader:
    def __init__(self, byte, max_length = 8):
        self.byte = byte
        self.position = 0
        self.max_length = max_length
    def set(self, byte, max_length):
        self.byte = byte
        self.position = 0
        self.max_length = max_length
    def __iter__(self):
        while self.position < self.max_length:
            shift = (8 - self.position - 1)
            mask = 1 << shift
            value = (self.byte & mask) >> shift
            yield value
            self.position += 1

class HuffmanDecoderUnicode:
    def __init__(self, tree: HuffmanTree, stream: io.BufferedReader):
        self.tree = tree
        self.stream = stream
        self.encoded_message_size = read_int32(stream)
        self.encoded_message_used_bits = read_int8(stream)
        self.current: Node = tree.root
    def read(self) -> Generator[str]:
        for i in range(self.encoded_message_size):
            length = 8 if i != self.encoded_message_size - 1 else self.encoded_message_used_bits
            byte = read_int8(self.stream)
            for bit in BitReader(byte, length):
                self.current = self.current.children[bit]
                if self.current.char:
                    yield self.current.char
                    self.current = self.tree.root
    @staticmethod
    def read_frequencies(stream: io.BufferedReader) -> List[tuple[str, int]]:
        frequency_table_length = read_int32(stream)
        frequencies = []
        for _ in range(frequency_table_length):
            char_len = read_int8(stream)
            char_bytes = stream.read(char_len)
            char = char_bytes.decode("utf-8")
            frequency = read_int32(stream)
            frequencies.append((char, frequency))
        return frequencies

def encode_streaming(input_path, output_path, chunk_size=8192):
    frequency_counter = Counter()
    file_size = pathlib.Path(input_path).stat().st_size
    with open(input_path, "r", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(chunk_size), ""):
            frequency_counter.update(chunk)
    tree = HuffmanTree.new(frequency_counter.most_common())
    with open(output_path, "wb") as fout:
        encoder = HuffmanEncoderUnicode(tree, fout)
        with open(input_path, "r", encoding="utf-8") as fin:
            with tqdm(total=file_size, desc="Encoding", unit="B", unit_scale=True) as pbar:
                for chunk in iter(lambda: fin.read(chunk_size), ""):
                    encoder.write_chunk(chunk)
                    pbar.update(len(chunk))
        encoder.close()

def assert_zhf_file_format(stream):
    file_format_constant = read_int32(stream)
    if file_format_constant != FILE_FORMAT_CONSTANT:
        raise ValueError("Invalid file format.")

def assert_file_format_version(stream):
    file_format_version = read_int8(stream)
    if file_format_version != FILE_FORMAT_VERSION:
        raise ValueError("Invalid file format version.")

def decode_streaming(input_path, output_path):
    with open(input_path, "rb") as fin:
        assert_zhf_file_format(fin)
        assert_file_format_version(fin)
        frequencies = HuffmanDecoderUnicode.read_frequencies(fin)
        tree = HuffmanTree.new(frequencies)
        decoder = HuffmanDecoderUnicode(tree, fin)
        with tqdm(total=decoder.encoded_message_size, desc="Decoding", unit="B", unit_scale=True) as pbar:
            with open(output_path, "w", encoding="utf-8") as fout:
                for char in decoder.read():
                    fout.write(char)
                    pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Huffman Encoder/Decoder CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode_parser = subparsers.add_parser("encode", help="Encode a text file into Huffman format (.zhf)")
    encode_parser.add_argument("--input", "-i", required=True, help="Input .txt file")
    encode_parser.add_argument("--output", "-o", required=True, help="Output .zhf file")
    decode_parser = subparsers.add_parser("decode", help="Decode a Huffman (.zhf) file into a .txt file")
    decode_parser.add_argument("--input", "-i", required=True, help="Input .zhf file")
    decode_parser.add_argument("--output", "-o", required=True, help="Output .txt file")
    args = parser.parse_args()
    if args.command == "encode":
        encode_streaming(args.input, args.output)
    elif args.command == "decode":
        decode_streaming(args.input, args.output)

if __name__ == "__main__":
    main()

def test_hello_world():
    encode_streaming("hello_world.txt", "hello_world.zhf")
    decode_streaming("hello_world.zhf", "hello_world_decoded.txt")
    assert pathlib.Path("hello_world.txt").read_text() == pathlib.Path("hello_world_decoded.txt").read_text()

def test_lorem_ipsum():
    encode_streaming("lorem.txt", "lorem.zhf")
    decode_streaming("lorem.zhf", "lorem_decoded.txt")
    assert pathlib.Path("lorem.txt").read_text() == pathlib.Path("lorem_decoded.txt").read_text()

def test_dust_and_circuits():
    encode_streaming("dust_and_circuits.txt", "dust_and_circuits.zhf")
    decode_streaming("dust_and_circuits.zhf", "dust_and_circuits_decoded.txt")
    assert pathlib.Path("dust_and_circuits.txt").read_text() == pathlib.Path("dust_and_circuits_decoded.txt").read_text()