package gafit

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSubmatrix(t *testing.T) {
	data := Dataset{
		X:        mat.NewDense(3, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
		ColNames: []string{"feat1", "feat2", "feat3"},
	}

	S := data.Submatrix([]string{"feat1", "feat3"})
	want := mat.NewDense(3, 2, []float64{1.0, 3.0, 4.0, 6.0, 7.0, 9.0})

	if !mat.EqualApprox(S, want, 1e-10) {
		t.Errorf("Want\n%v\ngot\n%v\n", want, S)
	}
}
