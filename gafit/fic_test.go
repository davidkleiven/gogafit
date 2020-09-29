package gafit

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFIC(t *testing.T) {
	fic := PredictionErrorFIC{
		Data: []int{0, 2},
	}

	X := mat.NewDense(3, 2, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	y := mat.NewVecDense(3, []float64{1.0, 2.0, 3.0})

	coeff := Fit(X, y)
	ficValue := fic.Evaluate(X, y, coeff)

	// Try to artificially change X for the first point and confirm fic is unchanged
	X.Set(1, 0, -0.5)
	ficAfter := fic.Evaluate(X, y, coeff)
	tol := 1e-6
	if math.Abs(ficValue-ficAfter) > tol {
		t.Errorf("FIC changed. Before: %f after %f\n", ficValue, ficAfter)
	}
}
