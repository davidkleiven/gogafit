package gafit

import (
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestRemove(t *testing.T) {
	for i, test := range []struct {
		data     []string
		toRemove string
		want     []string
	}{
		{
			data:     []string{"abc", "bcd", "q", "p"},
			toRemove: "bcd",
			want:     []string{"abc", "q", "p"},
		},
		{
			data:     []string{"abc", "bcd", "q", "p"},
			toRemove: "p",
			want:     []string{"abc", "bcd", "q"},
		},
		{
			data:     []string{"abc", "bcd", "q", "p"},
			toRemove: "pd",
			want:     []string{"abc", "bcd", "q", "p"},
		},
	} {
		data, _ := remove(test.data, test.toRemove)

		if !allEqualString(data, test.want) {
			t.Errorf("Test #%d: Wanted\n%v\ngot\n%v\n", i, test.want, data)
		}
	}
}

func TestParseValue(t *testing.T) {

	type wantType struct {
		values []float64
		err    error
	}
	for i, test := range []struct {
		data []string
		want wantType
	}{
		{
			data: []string{"6.5", "4.23", "5.67"},
			want: wantType{
				values: []float64{6.5, 4.23, 5.67},
				err:    nil,
			},
		},
	} {
		v, e := parseValues(test.data)

		if test.want.err != nil {
			if e != nil {
				t.Errorf("Test #%d: Expected error", i)
			}
		} else {
			if !floats.EqualApprox(test.want.values, v, 1e-10) {
				t.Errorf("Test #%d: Expected\n%v\ngot\n%v\n", i, test.want.values, v)
			}
		}
	}
}

func TestRemoveIndex(t *testing.T) {
	for i, test := range []struct {
		data []float64
		idx  int
		want []float64
	}{
		{
			data: []float64{2.0, 3.0, 4.0},
			idx:  1,
			want: []float64{2.0, 4.0},
		},
		{
			data: []float64{2.0, 3.0, 4.0},
			idx:  2,
			want: []float64{2.0, 3.0},
		},
	} {
		d, _ := removeIndex(test.data, test.idx)
		if !floats.EqualApprox(d, test.want, 1e-10) {
			t.Errorf("Test #%d: Wanted\n%v\ngot\n%v\n", i, test.want, d)
		}
	}
}

func TestReadFile(t *testing.T) {
	data, err := Read("_testdata/dataset.csv", "Var4")

	if err != nil {
		t.Errorf("Error during read %s\n", err)
		return
	}

	want := Dataset{
		ColNames:   []string{"Var1", "Var2", "Var3"},
		TargetName: "Var4",
		Y:          mat.NewVecDense(2, []float64{0.6, 0.7}),
		X:          mat.NewDense(2, 3, []float64{0.1, 0.3, 0.5, 0.2, 0.1, 0.5}),
	}

	if !data.IsEqual(want) {
		t.Errorf("Wanted\n%+v\ngot\n%+v\n", want, data)
	}
}
