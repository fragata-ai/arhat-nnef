
package core

import (
    "encoding/binary"
    "io"
)

//
//    Minor types and constants
//

const MaxRank = 8

type QuantCode int

const (
    QuantCodeFloat QuantCode = 0x00
    QuantCodeInteger = 0x01
    QuantCodeLinear = 0x10
    QuantCodeLogarithmic = 0x11
)

//
//    TensorHeader
//

type TensorHeader struct {
    Magic [2]uint8
    Version [2]uint8
    DataLength uint32
    Rank uint32
    Extents [MaxRank]uint32
    BitsPerItem uint32
    QuantCode uint32
    QuantParams [8]uint32
    Reserved [44]uint32
}

func(h *TensorHeader) Fill(
        version [2]int, 
        extents []int,
        bitsPerItem int,
        quantCode QuantCode) {
    const magic = "N\xEF"
    *h = TensorHeader{}
    h.Magic[0] = magic[0]
    h.Magic[1] = magic[1]
    h.Version[0] = uint8(version[0])
    h.Version[1] = uint8(version[1])
    rank := len(extents)
    if rank > MaxRank {
        RuntimeError("tensor rank %d exceeds maximum possible value (%d)", rank, MaxRank)
    }
    itemCount := 1
    for i := 0; i < rank; i++ {
        itemCount *= extents[i]
    }
    h.DataLength = uint32((itemCount * bitsPerItem + 7) / 8)
    h.BitsPerItem = uint32(bitsPerItem)
    h.Rank = uint32(rank)
    h.QuantCode = uint32(quantCode)
    for i := 0; i < rank; i++ {
        h.Extents[i] = uint32(extents[i])
    }
}

func(h *TensorHeader) Validate() {
    if h.Magic[0] != 'N' || h.Magic[1] != 0xEF {
        RuntimeError("invliad magic number in tensor binary")
    }
    if h.Version[0] != 1 || h.Version[1] != 0 {
        RuntimeError("unknown version number %d.%d", h.Version[0], h.Version[1])
    }
    if h.Rank > MaxRank {
        RuntimeError("tensor rank %d exceeds maximum allowed rank (%d)", h.Rank, MaxRank)
    }
    itemCount := uint32(1)
    for i := 0; i < int(h.Rank); i++ {
        itemCount *= h.Extents[i]
    }
    if h.DataLength != (itemCount * h.BitsPerItem + 7) / 8 {
        RuntimeError("data length is not compatible with extents and bits per item")
    }
    if h.QuantCode & 0xffff0000 == 0 { // Khronos-defined item type
        code := QuantCode(h.QuantCode & 0x0000ffff)
        switch code {
        case QuantCodeFloat:
            if h.BitsPerItem != 16 && h.BitsPerItem != 32 && h.BitsPerItem != 64 {
                RuntimeError("invalid bits per item for float item type: %d", h.BitsPerItem)
            }
        case QuantCodeInteger, QuantCodeLinear, QuantCodeLogarithmic:
            if h.BitsPerItem > 64 {
                RuntimeError("invalid bits per item for integer item type: %d", h.BitsPerItem)
            }
        default:
            RuntimeError("unkown Khronos-defined item type code: %x", code)
        }
    }
}

func(h *TensorHeader) Read(is io.Reader) {
    var err error
    var b [4]byte
    var w [2+MaxRank+2+8+11]uint32
    n, err := is.Read(b[:])
    if err != nil || n != len(b) {
        RuntimeError("invalid tensor header")
    }
    err = binary.Read(is, binary.LittleEndian, w[:])
    if err != nil {
        RuntimeError("invalid tensor header")
    }
    h.Magic[0] = b[0]
    h.Magic[1] = b[1]
    h.Version[0] = b[2]
    h.Version[1] = b[3]
    h.DataLength = w[0]
    h.Rank = w[1]
    copy(h.Extents[:], w[2:])
    pos := 2 + MaxRank
    h.BitsPerItem = w[pos]
    h.QuantCode = w[pos+1]
    copy(h.QuantParams[:], w[pos+2:])
    copy(h.Reserved[:], w[pos+10:])
}

func(h *TensorHeader) Write(os io.Writer) {
    var err error
    var b [4]byte
    var w [2+MaxRank+2+8+11]uint32
    b[0] = h.Magic[0]
    b[1] = h.Magic[1]
    b[2] = h.Version[0]
    b[3] = h.Version[1]
    w[0] = h.DataLength
    w[1] = h.Rank
    copy(w[2:], h.Extents[:])
    pos := 2 + MaxRank
    w[pos] = h.BitsPerItem
    w[pos+1] = h.QuantCode
    copy(w[pos+2:], h.QuantParams[:])
    copy(w[pos+10:], h.Reserved[:])
    _, err = os.Write(b[:])
    if err != nil {
        Raise(err)
    }
    err = binary.Write(os, binary.LittleEndian, w[:])
    if err != nil {
        Raise(err)
    }
}

//
//    Data exchange functions
//

func ReadScalarData(is io.Reader, bitsPerItem int, data []float32) {
    switch bitsPerItem {
    case 32:
        readData(is, data)
    case 64:
        count := len(data)
        temp := make([]float64, count)
        readData(is, temp)
        for i := 0; i < count; i++ {
            data[i] = float32(temp[i])
        }
    default:
        RuntimeError("cannot load float data of %d bits per item", bitsPerItem)
    }
}

func ReadIntegerData(is io.Reader, bitsPerItem int, data []int) {
    switch bitsPerItem {
    case 8:
        count := len(data)
        temp := make([]byte, count)
        readBytes(is, temp)
        for i := 0; i < count; i++ {
            data[i] = int(temp[i])
        }
    case 16:
        count := len(data)
        temp := make([]int16, count)
        readData(is, temp)
        for i := 0; i < count; i++ {
            data[i] = int(temp[i])
        }
    case 32:
        count := len(data)
        temp := make([]int32, count)
        readData(is, temp)
        for i := 0; i < count; i++ {
            data[i] = int(temp[i])
        }
    case 64:
        count := len(data)
        temp := make([]int64, count)
        readData(is, temp)
        for i := 0; i < count; i++ {
            data[i] = int(temp[i])
        }
    default:
        RuntimeError("cannot load int data of %d bits per item", bitsPerItem)
    }
}

func ReadLogicalData(is io.Reader, bitsPerItem int, data []bool) {
    switch bitsPerItem {
    case 1:
        count := len(data)
        temp := make([]byte, (count+7)/8)
        readBytes(is, temp)
        unpackBits(temp, data)
    case 8:
        count := len(data)
        temp := make([]byte, count)
        readBytes(is, temp)
        for i := 0; i < count; i++ {
            data[i] = (temp[i] != 0)
        }
    default:
        RuntimeError("cannot load bool data of %d bits per item", bitsPerItem)
    }
}

func WriteScalarData(os io.Writer, data []float32) {
    writeData(os, data)
}

func WriteIntegerData(os io.Writer, data []int) {
    count := len(data)
    temp := make([]int32, count)
    for i := 0; i < count; i++ {
        temp[i] = int32(data[i])
    }
    writeData(os, temp)
}

func WriteLogicalData(os io.Writer, data []bool) {
    count := len(data)
    temp := make([]byte, (count+7)/8)
    packBits(data, temp)
    writeBytes(os, temp)
}

func readBytes(is io.Reader, data []byte) {
    n, err := is.Read(data)
    if err != nil {
        Raise(err)
    }
    if n != len(data) {
        RuntimeError("Failed to read tensor data")
    }
}

func readData(is io.Reader, data interface{}) {
    err := binary.Read(is, binary.LittleEndian, data)
    if err != nil {
        Raise(err)
    }
}

func writeBytes(os io.Writer, data []byte) {
    _, err := os.Write(data)
    if err != nil {
        Raise(err)
    }
}

func writeData(os io.Writer, data interface{}) {
    err := binary.Write(os, binary.LittleEndian, data)
    if err != nil {
        Raise(err)
    }
}

func packBits(data []bool, buf []byte) {
    n := len(data)
    for i := 0; i < n; i++ {
        d := 0
        if data[i] {
            d = 1
        }
        buf[i/8] |= byte(d << uint(7 - (i % 8)))
    }
}

func unpackBits(buf []byte, data []bool) {
    n := len(data)
    for i := 0; i < n; i++ {
        d := (buf[i/8] >> uint(7 - (i % 8))) & 0x01
        data[i] = (d != 0)
    }
}

