package gafit

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNormalize(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1.0, 1.0, 1.0, 2.0})
	sq2 := 1.0 / math.Sqrt(2.0)
	sq5 := 1.0 / math.Sqrt(5.0)
	want := mat.NewDense(2, 2, []float64{sq2, sq5, sq2, 2.0 * sq5})
	normalize(X)

	if !mat.EqualApprox(X, want, 1e-6) {
		t.Errorf("Expected\n%v\ngot\n%v\n", want, X)
	}
}

func TestArgAbsMax(t *testing.T) {
	vec := mat.NewVecDense(5, []float64{1.0, 5.0, 4.0, -2.0, 1.0})
	want := 1
	max := argAbsMax(vec)

	if max != want {
		t.Errorf("Expected %d got%d\n", want, max)
	}
}

func TestSelection2BitString(t *testing.T) {
	selection := []int{2, 5, 3}
	want := []int{0, 0, 1, 1, 0, 1, 0}
	res := selection2bitstring(selection, 7)

	if !AllEqualInt(res, want) {
		t.Errorf("Expected\n%v\ngot\n%v\n", want, res)
	}
}

func TestGreadyOptimize(t *testing.T) {
	rows, cols := 50, 5
	X := mat.NewDense(rows, cols, nil)
	y := mat.NewVecDense(rows, nil)
	dx := 1.0 / float64(rows)
	for i := 0; i < rows; i++ {
		x := dx * float64(i)
		for j := 0; j < cols; j++ {
			X.Set(i, j, math.Pow(x, float64(j)))
		}
		y.SetVec(i, -2.0+4.0*x*x-0.5*x*x*x*x)
	}

	res := OrthogonalMatchingPursuit(X, y, Aicc, 8)

	// This model should be able to predict the result perfectly
	selected := []int{}
	for col, i := range res.Include {
		if i == 1 {
			selected = append(selected, col)
		}
	}
	sub := subMatrix(X, selected)
	rmse := Rmse(sub, y, res.Coeff)
	if rmse > 1e-10 {
		t.Errorf("Model does not predict perfectly\n")
	}
}
