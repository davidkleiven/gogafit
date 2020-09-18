package gafit

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFit(t *testing.T) {
	tol := 1e-8
	for i, test := range []struct {
		X      *mat.Dense
		Y      *mat.VecDense
		Expect *mat.VecDense
	}{
		// Case 1: fully determined
		{
			X:      mat.NewDense(2, 2, []float64{1.0, 0.0, 1.0, 1.0}),
			Y:      mat.NewVecDense(2, []float64{0.0, 1.0}),
			Expect: mat.NewVecDense(2, []float64{0.0, 1.0}),
		},
		// Case 2: over determined
		{
			X:      mat.NewDense(3, 2, []float64{1.0, 0.0, 1.0, 1.0, 1.0, 2.0}),
			Y:      mat.NewVecDense(3, []float64{0.0, 1.0, 2.0}),
			Expect: mat.NewVecDense(2, []float64{0.0, 1.0}),
		},
		// Case 3: under determined
		{
			X:      mat.NewDense(1, 2, []float64{1.0, 1.0}),
			Y:      mat.NewVecDense(1, []float64{1.0}),
			Expect: mat.NewVecDense(2, []float64{0.5, 0.5}),
		},
	} {
		coeff := Fit(test.X, test.Y)
		if !mat.EqualApprox(coeff, test.Expect, tol) {
			t.Errorf("Test #%d: Expected\n%v\nGot\n%v\n", i, test.Expect, coeff)
		}
	}
}

func TestPred(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0})
	c := mat.NewVecDense(2, []float64{1.0, -2.0})
	got := Pred(X, c)
	want := mat.NewVecDense(2, []float64{-3.0, -5.0})
	tol := 1e-8
	if !mat.EqualApprox(got, want, tol) {
		t.Errorf("Expected\n%v\nGot\n%v\n", mat.Formatted(want), mat.Formatted(got))
	}
}

func TestMatrixEqual(t *testing.T) {
	for i, test := range []struct {
		x    *mat.Dense
		y    *mat.Dense
		want bool
	}{
		{
			x:    nil,
			y:    nil,
			want: true,
		},
		{
			x:    nil,
			y:    mat.NewDense(2, 1, []float64{1.0, 2.0}),
			want: false,
		},
		{
			x:    mat.NewDense(2, 1, []float64{1.0, 2.0}),
			y:    nil,
			want: false,
		},
		{
			x:    mat.NewDense(2, 1, []float64{1.0, 2.0}),
			y:    mat.NewDense(2, 1, []float64{1.0, 2.0}),
			want: true,
		},
		{
			x:    mat.NewDense(3, 1, []float64{1.0, 2.0, 2.0}),
			y:    mat.NewDense(2, 1, []float64{1.0, 2.0}),
			want: false,
		},
		{
			x:    mat.NewDense(2, 1, []float64{1.0, 1.0}),
			y:    mat.NewDense(2, 1, []float64{1.0, 2.0}),
			want: false,
		},
	} {
		got := matrixEqual(test.x, test.y, 1e-6)
		if got != test.want {
			t.Errorf("Test #%d: Got %v want %v\n", i, got, test.want)
		}
	}
}

func TestVectorEqual(t *testing.T) {
	for i, test := range []struct {
		x    *mat.VecDense
		y    *mat.VecDense
		want bool
	}{
		{
			x:    nil,
			y:    nil,
			want: true,
		},
		{
			x:    nil,
			y:    mat.NewVecDense(2, []float64{1.0, 2.0}),
			want: false,
		},
		{
			x:    mat.NewVecDense(2, []float64{1.0, 2.0}),
			y:    nil,
			want: false,
		},
		{
			x:    mat.NewVecDense(2, []float64{1.0, 2.0}),
			y:    mat.NewVecDense(2, []float64{1.0, 2.0}),
			want: true,
		},
		{
			x:    mat.NewVecDense(3, []float64{1.0, 2.0, 2.0}),
			y:    mat.NewVecDense(2, []float64{1.0, 2.0}),
			want: false,
		},
		{
			x:    mat.NewVecDense(2, []float64{1.0, 1.0}),
			y:    mat.NewVecDense(2, []float64{1.0, 2.0}),
			want: false,
		},
	} {
		got := vectorEqual(test.x, test.y, 1e-6)
		if got != test.want {
			t.Errorf("Test #%d: Got %v want %v\n", i, got, test.want)
		}
	}
}
