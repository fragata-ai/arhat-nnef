//
// Copyright (c) 2019-2020 FRAGATA COMPUTER SYSTEMS AG
// Copyright (c) 2017 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

//
// Based on the code of The Khronos Group Inc. NNEF Tools.
// Ported from C++ to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package comp

import (
    "fmt"
    "fragata/arhat/nnef/core"
)

//
//    Assignment
//

type Assignment struct {
    lhs Expr
    rhs Expr
}

func NewAssignment(lhs Expr, rhs Expr) *Assignment {
    a := new(Assignment)
    a.Init(lhs, rhs)
    return a
}

func(a *Assignment) Init(lhs Expr, rhs Expr) {
    a.lhs = lhs
    a.rhs = rhs
}

func(a *Assignment) Lhs() Expr {
    return a.lhs
}

func(a *Assignment) Rhs() Expr {
    return a.rhs
}

func(a *Assignment) String() string {
    return fmt.Sprintf("%s = %s", a.lhs.String(), a.rhs.String())
}

//
//    Fragment
//

type Fragment struct {
    prototype *core.Prototype
    assignments []*Assignment
}

func NewFragment(prototype *core.Prototype, assignments []*Assignment) *Fragment {
    f := new(Fragment)
    f.Init(prototype, assignments)
    return f
}

func(f *Fragment) Init(prototype *core.Prototype, assignments []*Assignment) {
    f.prototype = prototype
    f.assignments = assignments
}

func(f *Fragment) Prototype() *core.Prototype {
    return f.prototype
}

func(f *Fragment) AssignmentCount() int {
    return len(f.assignments)
}

func(f *Fragment) AssignmentAt(idx int) *Assignment {
    return f.assignments[idx]
}

func(f *Fragment) String() string {
    s := f.prototype.String() + "\n"
    n := len(f.assignments)
    if n != 0 {
        s += "{\n"
        for i := 0; i < n; i++ {
            s += "    " + f.assignments[i].String() + "\n"
        }
        s += "}\n"
    }
    return s
}

