cdef class BufferedBitWriterNative:
    cdef object stream  # Remove type annotation here
    cdef bytearray byte_buffer
    cdef int bit_buffer
    cdef int bit_pos
    cdef int buffer_size

    def __cinit__(self, stream, int buffer_size=4096):  # Removed type annotation
        self.stream = stream
        self.byte_buffer = bytearray()
        self.bit_buffer = 0
        self.bit_pos = 0
        self.buffer_size = buffer_size

    @property
    def bit_pos(self):
        return self.bit_pos

    cpdef void write_bit(self, int bit):
        self.bit_buffer = (self.bit_buffer << 1) | (bit & 1)
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.byte_buffer.append(self.bit_buffer & 0xFF)
            self.bit_buffer = 0
            self.bit_pos = 0
            if len(self.byte_buffer) >= self.buffer_size:
                self.flush_bytes()

    cpdef void write_bits(self, unsigned int bits, int length):
        cdef int i
        for i in range(length - 1, -1, -1):
            self.write_bit((bits >> i) & 1)

    cpdef void flush_bytes(self):
        if self.byte_buffer:
            self.stream.write(self.byte_buffer)
            self.byte_buffer.clear()

    cpdef void flush(self):
        if self.bit_pos > 0:
            self.bit_buffer <<= (8 - self.bit_pos)
            self.byte_buffer.append(self.bit_buffer & 0xFF)
            self.bit_buffer = 0
            self.bit_pos = 0
        self.flush_bytes()

    cpdef int bytes_written(self):
        return len(self.byte_buffer) * 8 + self.bit_pos