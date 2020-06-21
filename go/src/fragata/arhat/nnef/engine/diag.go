
package engine

import (
    "fmt"
    "math"
    "fragata/arhat/nnef/core"
    "fragata/arhat/nnef/runtime"
)

//
//    Diagnostic functions
//

func checkFinite(graph *core.Graph, ctx *runtime.Context, step int, op *core.Operation) {
    count := op.OutputCount()
    for i := 0; i < count; i++ {
        output := op.OutputAt(i)
        name := output.Identifier()
        tensor := graph.GetTensor(name)
        tensor.ResizeData()
        ctx.ReadTensor(tensor)
        if tensor.Dtype() == "scalar" {
            data := tensor.ScalarData()
            posInfCount := 0
            negInfCount := 0
            nanCount := 0
            for _, value := range data {
                switch {
                case math.IsInf(float64(value), 1):
                    posInfCount++
                case math.IsInf(float64(value), -1):
                    negInfCount++
                case math.IsNaN(float64(value)):
                    nanCount++
                }
            }
            fmt.Printf("Step %d, op %s, output %d, tensor %s: +Inf %d -Inf %d NaN %d out of %d\n",
                step, op.Name(), i, name, posInfCount, negInfCount, nanCount, len(data))
        }
    }
}

