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

func TestColumns(t *testing.T) {
	data := Dataset{
		X:        mat.NewDense(3, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
		ColNames: []string{"feat1", "feat2", "feat3"},
	}

	for i, test := range []struct {
		pattern string
		want    []int
	}{
		{
			pattern: "fe",
			want:    []int{0, 1, 2},
		},
		{
			pattern: "1",
			want:    []int{0},
		},
		{
			pattern: "qu",
			want:    []int{},
		},
		{
			pattern: "eat",
			want:    []int{0, 1, 2},
		},
	} {
		cols := data.Columns(test.pattern)
		if !AllEqualInt(cols, test.want) {
			t.Errorf("Test #%d: Want\n%v\ngot\n%v\n", i, test.want, cols)
		}
	}
}

func TestCopy(t *testing.T) {
	data := Dataset{
		X:        mat.NewDense(3, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
		Y:        mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
		ColNames: []string{"feat1", "feat2", "feat3"},
	}

	dataCpy := data.Copy()

	if !data.IsEqual(dataCpy) {
		t.Errorf("Copy does not match\n")
	}
}

func TestAddPoly(t *testing.T) {
	data := Dataset{
		X:        mat.NewDense(3, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
		ColNames: []string{"feat1", "feat2", "feat3"},
	}

	type wantType struct {
		X        *mat.Dense
		ColNames []string
	}

	for i, test := range []struct {
		cols  []int
		order int
		want  wantType
	}{
		{
			cols:  []int{},
			order: 1,
			want: wantType{
				X:        mat.DenseCopyOf(data.X),
				ColNames: []string{"feat1", "feat2", "feat3"},
			},
		},
		{
			cols:  []int{},
			order: 2,
			want: wantType{
				X:        mat.DenseCopyOf(data.X),
				ColNames: []string{"feat1", "feat2", "feat3"},
			},
		},
		{
			cols:  []int{0},
			order: 2,
			want: wantType{
				X: mat.NewDense(3, 4, []float64{1.0, 2.0, 3.0, 1.0,
					4.0, 5.0, 6.0, 16.0,
					7.0, 8.0, 9.0, 49.0}),
				ColNames: []string{"feat1", "feat2", "feat3", "feat1p2"},
			},
		},
		{
			cols:  []int{0},
			order: 3,
			want: wantType{
				X: mat.NewDense(3, 5, []float64{1.0, 2.0, 3.0, 1.0, 1.0,
					4.0, 5.0, 6.0, 16.0, 64.0,
					7.0, 8.0, 9.0, 49.0, 343.0}),
				ColNames: []string{"feat1", "feat2", "feat3", "feat1p2", "feat1p3"},
			},
		},
		{
			cols:  []int{0, 2},
			order: 2,
			want: wantType{
				X: mat.NewDense(3, 5, []float64{1.0, 2.0, 3.0, 1.0, 9.0,
					4.0, 5.0, 6.0, 16.0, 36.0,
					7.0, 8.0, 9.0, 49.0, 81.0}),
				ColNames: []string{"feat1", "feat2", "feat3", "feat1p2", "feat3p2"},
			},
		},
		{
			cols:  []int{0, 2},
			order: 3,
			want: wantType{
				X: mat.NewDense(3, 7, []float64{1.0, 2.0, 3.0, 1.0, 1.0, 9.0, 27.0,
					4.0, 5.0, 6.0, 16.0, 64.0, 36.0, 216.0,
					7.0, 8.0, 9.0, 49.0, 343.0, 81.0, 729.0}),
				ColNames: []string{"feat1", "feat2", "feat3", "feat1p2", "feat1p3", "feat3p2", "feat3p3"},
			},
		},
	} {
		res := AddPoly(test.cols, data, test.order)

		if !mat.EqualApprox(res.X, test.want.X, 1e-8) {
			t.Errorf("Test #%d: Want\n%v\ngot\n%v\n", i, mat.Formatted(test.want.X), mat.Formatted(res.X))
		}

		if !allEqualString(test.want.ColNames, res.ColNames) {
			t.Errorf("Test #%d: Want\n%v\ngot\n%v\n", i, test.want.ColNames, res.ColNames)
		}
	}
}
