
package main

func TopK(data []float32, k int) ([]int, []float32) {
    pos := make([]int, k)
    val := make([]float32, k)
    for i := 0; i < k; i++ {
        pos[i] = -1
        val[i] = 0.0
    }
    count := len(data)
    for p := 0; p < count; p++ {
        v := data[p]
        j := -1
        for i := 0; i < k; i++ {
            if pos[i] < 0 || val[i] < v {
                j = i;
                break;
            }
        }
        if j >= 0 {
            for i := k - 1; i > j; i-- {
                pos[i] = pos[i-1]
                val[i] = val[i-1]
            }
            pos[j] = p
            val[j] = v
        }
    }
    return pos, val
}

