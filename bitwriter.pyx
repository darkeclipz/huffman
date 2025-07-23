# bitwriter.pyx
# cython: language_level=3
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t
cimport cython

cdef class BufferedBitWriterNative:
    cdef:
        object stream               # Python buffered writer
        bytearray byte_buffer       # Buffer bytes before writing to stream
        uint8_t bit_buffer          # Accumulate bits here
        int bit_pos                 # Number of bits currently in bit_buffer (0-7)
        int buffer_size             # Flush after this many bytes in byte_buffer

    def __cinit__(self, stream, int buffer_size=4096):
        self.stream = stream
        self.byte_buffer = bytearray()
        self.bit_buffer = 0
        self.bit_pos = 0
        self.buffer_size = buffer_size

    @property
    def bit_pos(self):
        return self.bit_pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef write_bit(self, int bit):
        # bit must be 0 or 1
        self.bit_buffer = (self.bit_buffer << 1) | (bit & 1)
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.byte_buffer.append(self.bit_buffer)
            self.bit_buffer = 0
            self.bit_pos = 0
            if len(self.byte_buffer) >= self.buffer_size:
                self.flush_bytes()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef write_chunk(self, text, dict encoding_table):
        """
        text: Python str (Unicode)
        encoding_table: dict mapping character (str) -> Node with
                        Node.encoding_path = (int path, int length)
        """
        cdef Py_ssize_t i, n = len(text)
        cdef object ch
        cdef tuple path_info
        cdef int path, steps, bit_index, bit_val

        for i in range(n):
            ch = text[i]
            path_info = encoding_table[ch].encoding_path
            path = path_info[0]
            steps = path_info[1]

            # write bits from most significant to least significant
            for bit_index in range(steps - 1, -1, -1):
                bit_val = (path >> bit_index) & 1
                self.write_bit(bit_val)

    cpdef flush_bytes(self):
        if self.byte_buffer:
            self.stream.write(self.byte_buffer)
            self.byte_buffer.clear()

    cpdef flush(self):
        if self.bit_pos > 0:
            # pad remaining bits with zeros on the right
            self.bit_buffer <<= (8 - self.bit_pos)
            self.byte_buffer.append(self.bit_buffer)
            self.bit_pos = 0
        self.flush_bytes()

    cpdef int bits_written(self):
        return len(self.byte_buffer) * 8 + self.bit_pos
