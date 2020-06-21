
package reference

//
//    NdLoop2
//

type NdLoop2 struct {
    shape [2]int
    index [2]int
    test bool
}

func(l *NdLoop2) Start(shape []int) {
    l.shape[0] = shape[0]
    l.shape[1] = shape[1]
    l.index[0] = 0
    l.index[1] = 0
    l.test = true
}

func(l *NdLoop2) Test() bool {
    return l.test
}

func(l *NdLoop2) Next() {
    l.index[1]++
    if  l.index[1] < l.shape[1] {
        return
    }
    l.index[1] = 0
    l.index[0]++
    if  l.index[0] < l.shape[0] {
        return
    }
    l.test = false
}

func(l *NdLoop2) Index() []int {
    return l.index[:]
}

func NdOffset2(shape []int, index []int) int {
    offset := index[0]
    offset = offset * shape[1] + index[1]
    return offset
}

//
//    NdLoop3
//

type NdLoop3 struct {
    shape [3]int
    index [3]int
    test bool
}

func(l *NdLoop3) Start(shape []int) {
    l.shape[0] = shape[0]
    l.shape[1] = shape[1]
    l.shape[2] = shape[2]
    l.index[0] = 0
    l.index[1] = 0
    l.index[2] = 0
    l.test = true
}

func(l *NdLoop3) Test() bool {
    return l.test
}

func(l *NdLoop3) Next() {
    l.index[2]++
    if  l.index[2] < l.shape[2] {
        return
    }
    l.index[2] = 0
    l.index[1]++
    if  l.index[1] < l.shape[1] {
        return
    }
    l.index[1] = 0
    l.index[0]++
    if  l.index[0] < l.shape[0] {
        return
    }
    l.test = false
}

func(l *NdLoop3) Index() []int {
    return l.index[:]
}

func NdOffset3(shape []int, index []int) int {
    offset := index[0]
    offset = offset * shape[1] + index[1]
    offset = offset * shape[2] + index[2]
    return offset
}

//
//    NdLoop4
//

type NdLoop4 struct {
    shape [4]int
    index [4]int
    test bool
}

func(l *NdLoop4) Start(shape []int) {
    l.shape[0] = shape[0]
    l.shape[1] = shape[1]
    l.shape[2] = shape[2]
    l.shape[3] = shape[3]
    l.index[0] = 0
    l.index[1] = 0
    l.index[2] = 0
    l.index[3] = 0
    l.test = true
}

func(l *NdLoop4) Test() bool {
    return l.test
}

func(l *NdLoop4) Next() {
    l.index[3]++
    if  l.index[3] < l.shape[3] {
        return
    }
    l.index[3] = 0
    l.index[2]++
    if  l.index[2] < l.shape[2] {
        return
    }
    l.index[2] = 0
    l.index[1]++
    if  l.index[1] < l.shape[1] {
        return
    }
    l.index[1] = 0
    l.index[0]++
    if  l.index[0] < l.shape[0] {
        return
    }
    l.test = false
}

func(l *NdLoop4) Index() []int {
    return l.index[:]
}

func NdOffset4(shape []int, index []int) int {
    offset := index[0]
    offset = offset * shape[1] + index[1]
    offset = offset * shape[2] + index[2]
    offset = offset * shape[3] + index[3]
    return offset
}

//
//    NdLoop5
//

type NdLoop5 struct {
    shape [5]int
    index [5]int
    test bool
}

func(l *NdLoop5) Start(shape []int) {
    l.shape[0] = shape[0]
    l.shape[1] = shape[1]
    l.shape[2] = shape[2]
    l.shape[3] = shape[3]
    l.shape[4] = shape[4]
    l.index[0] = 0
    l.index[1] = 0
    l.index[2] = 0
    l.index[3] = 0
    l.index[4] = 0
    l.test = true
}

func(l *NdLoop5) Test() bool {
    return l.test
}

func(l *NdLoop5) Next() {
    l.index[4]++
    if  l.index[4] < l.shape[4] {
        return
    }
    l.index[4] = 0
    l.index[3]++
    if  l.index[3] < l.shape[3] {
        return
    }
    l.index[3] = 0
    l.index[2]++
    if  l.index[2] < l.shape[2] {
        return
    }
    l.index[2] = 0
    l.index[1]++
    if  l.index[1] < l.shape[1] {
        return
    }
    l.index[1] = 0
    l.index[0]++
    if  l.index[0] < l.shape[0] {
        return
    }
    l.test = false
}

func(l *NdLoop5) Index() []int {
    return l.index[:]
}

func NdOffset5(shape []int, index []int) int {
    offset := index[0]
    offset = offset * shape[1] + index[1]
    offset = offset * shape[2] + index[2]
    offset = offset * shape[3] + index[3]
    offset = offset * shape[4] + index[4]
    return offset
}

