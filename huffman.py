from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Generator, List
import io
import pathlib
from collections import Counter
import argparse

FILE_FORMAT_CONSTANT = 0x5A4846
FILE_FORMAT_VERSION = 1

@dataclass
class Node:
    id: int
    parent: Node | None
    character: str | None
    weight: int
    children: List[Node]
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

class BitBuffer:
    write_bit_buffer: int = 0
    write_bit_position: int = 0
    def write_bit(self, stream: io.BufferedWriter, value: int):
        assert value == 0 or value == 1
        mask = value << (8 - self.write_bit_position - 1)
        self.write_bit_buffer |= mask
        self.write_bit_position += 1
        if self.write_bit_position >= 8:
            write_int8(stream, self.write_bit_buffer)
            self.write_bit_buffer = 0
            self.write_bit_position = 0
    def flush_bit_buffer(self, stream: io.BufferedWriter):
        if self.write_bit_position != 0:
            write_int8(stream, self.write_bit_buffer)

def write_int8(stream: io.BufferedWriter, value: int) -> None:
    stream.write(value.to_bytes(1, "big"))

def write_int32(stream: io.BufferedWriter, value: int) -> None:
    stream.write(value.to_bytes(4, "big"))

def read_int8(stream: io.BufferedReader) -> int:
    return int.from_bytes(stream.read(1), "big")

def read_int32(stream: io.BufferedReader) -> int:
    return int.from_bytes(stream.read(4), "big")

def pop_min_node(nodes: List[Node]) -> Node:
    min_index = 0
    min_weight = 1e99
    for i, node in enumerate(nodes):
        if node.weight < min_weight:
            min_index = i
            min_weight = node.weight
    return nodes.pop(min_index)

class HuffmanTree:
    def __init__(self, root: Node, encoding_table: dict[str, Node]):
        self.root = root
        self.encoding_table = encoding_table
    @staticmethod
    def new(frequency_list: List[tuple[str, int]]):
        nodes = []
        encoding_table: dict[str, Node] = {}
        for character, occurance in frequency_list:
            node = Node.new_leaf(character, occurance)
            nodes.append(node)
            encoding_table[character] = node
        while len(nodes) >= 2:
            last_node = pop_min_node(nodes)
            second_last_node = pop_min_node(nodes)
            merged_weight = last_node.weight + second_last_node.weight
            merged_node = Node.new_node(merged_weight, left=second_last_node, right=last_node)
            last_node.parent = merged_node
            second_last_node.parent = merged_node
            nodes.append(merged_node)
        return HuffmanTree(nodes[0], encoding_table)

class HuffmanEncoder:
    def __init__(self, tree: HuffmanTree, stream: io.BufferedWriter):
        self.tree = tree
        self.stream = stream
        self.write_header()
        self.bit_buffer = BitBuffer()
    def write_header(self):
        write_int32(self.stream, FILE_FORMAT_CONSTANT)
        write_int8(self.stream, FILE_FORMAT_VERSION)
        encoding_table_size = len(self.tree.encoding_table)
        write_int32(self.stream, encoding_table_size)
        for character, node in self.tree.encoding_table.items():
            write_int8(self.stream, ord(character))
            write_int32(self.stream, node.weight)
        self.encoded_message_size_position = self.stream.tell()
        write_int32(self.stream, 0)
        self.encoded_message_used_bits_position = self.stream.tell()
        write_int8(self.stream, 0)
    def write(self, character) -> None:
        current: Node = self.tree.encoding_table[character]
        path: List[int] = []
        while current.parent:
            previous = current
            current = current.parent
            bit = current.children.index(previous)
            path.append(bit)
        while path:
            self.bit_buffer.write_bit(self.stream, path.pop())
    def close(self):
        self.bit_buffer.flush_bit_buffer(self.stream)
        total_bytes_written = self.stream.tell() - self.encoded_message_used_bits_position - 1
        self.stream.seek(self.encoded_message_size_position)
        write_int32(self.stream, total_bytes_written)
        self.stream.seek(self.encoded_message_used_bits_position)
        global write_bit_position
        write_int8(self.stream, self.bit_buffer.write_bit_position)
        self.stream.seek(0)

class BitReader:
    def __init__(self, byte, max_length = 8):
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

class HuffmanDecoder:
    def __init__(self, tree: HuffmanTree, stream: io.BufferedReader):
        self.tree = tree
        self.stream = stream
        self.encoded_message_size = read_int32(stream)
        self.encoded_message_used_bits = read_int8(stream)
        self.current_node = tree.root
    def read(self) -> Generator[str]:
        for i in range(self.encoded_message_size):
            length = 8 if i != self.encoded_message_size - 1 else self.encoded_message_used_bits
            byte = read_int8(self.stream)
            for bit in BitReader(byte, length):
                self.current_node = self.current_node.children[bit]
                if self.current_node.character:
                    yield self.current_node.character
                    self.current_node = self.tree.root

def assert_ascii_text(text):
    if not text.isascii():
            raise ValueError("Input file must contain only ASCII characters. Unicode is not supported.")
    
def encode_streaming(input_path, output_path, chunk_size=8192):
    frequency_counter = Counter()
    with open(input_path, "r", encoding="ascii") as f:
        for chunk in iter(lambda: f.read(chunk_size), ""):
            if not chunk.isascii():
                raise ValueError("Input must be ASCII only.")
            frequency_counter.update(chunk)
    tree = HuffmanTree.new(frequency_counter.most_common())
    with open(output_path, "wb") as fout:
        encoder = HuffmanEncoder(tree, fout)
        with open(input_path, "r", encoding="ascii") as fin:
            for chunk in iter(lambda: fin.read(chunk_size), ""):
                for char in chunk:
                    encoder.write(char)
        encoder.close()

def assert_zhf_file_format(stream):
    file_format_constant = read_int32(stream)
    if file_format_constant != FILE_FORMAT_CONSTANT:
        raise ValueError("Invalid file format.")
    
def assert_file_format_version(stream):
    file_format_version = read_int8(stream)
    if file_format_version != FILE_FORMAT_VERSION:
        raise ValueError("Invalid file format version.")

def read_frequencies(stream: io.BufferedReader) -> List[tuple[str, int]]:
    frequency_table_length = read_int32(stream)
    frequencies = []
    for _ in range(frequency_table_length):
        character = chr(read_int8(stream))
        occurances = read_int32(stream)
        frequencies.append((character, occurances))
    return frequencies

def decode_streaming(input_path, output_path):
    with open(input_path, "rb") as fin:
        assert_zhf_file_format(fin)
        assert_file_format_version(fin)
        frequencies = read_frequencies(fin)
        tree = HuffmanTree.new(frequencies)
        decoder = HuffmanDecoder(tree, fin)
        with open(output_path, "w", encoding="ascii") as fout:
            for char in decoder.read():
                fout.write(char)

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