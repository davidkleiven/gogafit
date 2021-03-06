package gafit

import (
	"math"
	"math/rand"
	"os"
	"reflect"
	"testing"

	"github.com/MaxHalford/eaopt"
	"gonum.org/v1/gonum/mat"
)

func TestFit(t *testing.T) {
	tol := 1e-7
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

func TestRmse(t *testing.T) {
	X := mat.NewDense(10, 4, nil)
	y := mat.NewVecDense(10, nil)
	src := rand.NewSource(42)
	rng := rand.New(src)
	std := 0.1
	for i := 0; i < 10; i++ {
		for j := 0; j < 4; j++ {
			X.Set(i, j, math.Pow(0.1*float64(i), float64(j)))
		}
		y.SetVec(i, 5.0*X.At(i, 0)-2.0*X.At(i, 2)+std*rng.NormFloat64())
	}

	coeff := mat.NewVecDense(4, []float64{5.0, 0.0, -2.0, 0.0})
	rmse := Rmse(X, y, coeff)

	// RMSE should be close to the std
	f := 0.05
	if math.Abs(rmse-std) > f*std {
		t.Errorf("Expected %f +- %f. Got %f\n", std, f*std, rmse)
	}
}

func TestHatMatrix(t *testing.T) {
	for i, test := range []struct {
		X    *mat.Dense
		want mat.Matrix
	}{
		// When X is square, the hat matrix should be equal to the identity matrix
		{
			X:    mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
			want: mat.NewDiagDense(2, []float64{1.0, 1.0}),
		},
		// Num cols larger than num rows, hat matrix should be identify
		{
			X:    mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
			want: mat.NewDiagDense(2, []float64{1.0, 1.0}),
		},
		// Num rows larger than num cols. Expected hat matrix calculated with Numpy
		{
			X: mat.NewDense(3, 2, []float64{1.0, 0.0, 1.0, 1.0, 1.0, 2.0}),
			want: mat.NewDense(3, 3, []float64{0.83333333, 0.33333333, -0.16666667,
				0.33333333, 0.33333333, 0.33333333,
				-0.16666667, 0.33333333, 0.83333333}),
		},
	} {
		H := HatMatrix(test.X)

		if !mat.EqualApprox(H, test.want, 1e-8) {
			t.Errorf("Test #%d: Want\n%v\ngot\n%v\n", i, mat.Formatted(test.want), mat.Formatted(H))
		}
	}
}

func TestSubmatrixView(t *testing.T) {
	sub := SubMatrix{
		X:    mat.NewDense(3, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
		Rows: 2,
		Cols: 3,
	}

	r, c := sub.Dims()
	if (r != 2) || (c != 3) {
		t.Errorf("Expected (2, 3) got (%d, %d)\n", r, c)
	}

	want := mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	got := mat.NewDense(2, 3, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			got.Set(i, j, sub.At(i, j))
		}
	}

	if !mat.EqualApprox(want, got, 1e-10) {
		t.Errorf("Want\n%v\ngot\n%v\n", want, got)
	}

	subT := sub.T()

	if !mat.EqualApprox(subT, want.T(), 1e-10) {
		t.Errorf("Want\n%v\ngot\n%v\n", want.T(), subT)
	}
}

func TestCovarianceMatrix(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1.0, 1.0, 1.0, 2.0, 1.0, 3.0})
	y := mat.NewVecDense(3, []float64{-0.2, -0.8, -5.0})
	coeff := Fit(X, y)
	rss := Rss(X, y, coeff)
	cov, _ := CovMatrix(X, rss)

	// Expected covariance matrix
	sigma := rss / (3.0 - 2.0)
	meanX := mat.Sum(X.ColView(1)) / 3.0
	meanXSq := (1.0 + 4.0 + 9.0) / 3.0

	devSq := 0.0
	for i := 0; i < 3; i++ {
		devSq += math.Pow(X.At(i, 1)-meanX, 2.0)
	}

	varBeta0 := sigma * meanXSq / devSq
	varBeta1 := sigma * 1.0 / devSq
	cov01 := -sigma * meanX / devSq
	expectCov := mat.NewDense(2, 2, []float64{varBeta0, cov01, cov01, varBeta1})

	if !mat.EqualApprox(cov, expectCov, 1e-6) {
		t.Errorf("Expected\n%v\ngot\n%v\n", mat.Formatted(expectCov), mat.Formatted(cov))
	}
}

func TestJoin2Map(t *testing.T) {
	keys := []string{"abc", "def"}
	values := []float64{1.0, 2.0}
	res := join2map(keys, values)
	want := make(map[string]float64)
	want["abc"] = 1.0
	want["def"] = 2.0

	if !reflect.DeepEqual(res, want) {
		t.Errorf("Expected\n%v\ngot\n%v\n", want, res)
	}
}

func TestGetPredictions(t *testing.T) {
	dataset := Dataset{
		X:          mat.NewDense(3, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
		Y:          mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
		ColNames:   []string{"feat1", "feat2", "feat3"},
		TargetName: "mytarget",
	}

	model := Model{
		Coeffs: make(map[string]float64),
	}
	model.Coeffs["feat3"] = -2.0
	model.Coeffs["feat1"] = 1.0

	coeffs := mat.NewVecDense(3, []float64{1.0, 0.0, -2.0})
	pred := Pred(dataset.X, coeffs)
	predictions := GetPredictions(dataset, model, nil)

	// Check that all agree
	tol := 1e-8
	for i := 0; i < 3; i++ {
		if math.Abs(predictions[i].Value-pred.AtVec(i)) > tol {
			t.Errorf("Expected %f got %f\n", pred.AtVec(i), predictions[i].Value)
		}
	}
}

func TestSaveReadRoundTrip(t *testing.T) {
	predOrig := []Prediction{
		{
			Value: 1.0,
			Std:   2.0,
		},
		{
			Value: -2.0,
			Std:   0.01,
		},
		{
			Value: 3.0,
			Std:   0.05,
		},
	}

	outfile := "predDemo.csv"
	defer os.Remove(outfile)
	err := SavePredictions(outfile, predOrig)
	if err != nil {
		t.Errorf("Error during save: %s\n", err)
		return
	}

	predRead, err := ReadPredictions(outfile)
	if err != nil {
		t.Errorf("Error during read: %s\n", err)
		return
	}

	if len(predRead) != len(predOrig) {
		t.Errorf("Lentgh differ. Expected %d got %d\n", len(predOrig), len(predRead))
		return
	}
	for i := 0; i < len(predOrig); i++ {
		if !predOrig[i].IsEqual(predRead[i]) {
			t.Errorf("Expected\n%+v\ngot\n%+v\n", predOrig[i], predRead[i])
		}
	}

}

func TestCovarianceConsistency(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{1.0, 2.0, -3.0, 4.0, 5.0, 6.7, 7.0, 8.0, 9.0, 10.0, -11.0, 12.0})
	hat := HatMatrix(X)
	cov, err := CovMatrix(X, 1.0)
	if err != nil {
		t.Errorf("Error occured in CovMatrix: %s\n", err)
		return
	}

	// Hat matrix should be X*cov*X.T
	hatFromCov := mat.NewDense(4, 4, nil)
	hatFromCov.Product(X, cov, X.T())
	if !mat.EqualApprox(hatFromCov, hat, 1e-8) {
		t.Errorf("Expected\n%v\ngot\n%v\n", mat.Formatted(hat), mat.Formatted(hatFromCov))
	}
}

func TestGADefaultLogger(t *testing.T) {
	ga, err := eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		t.Errorf("Could not initialize ga: %s\n", err)
	}
	ga.Callback = GAProgressLogger
	factory := LinearModelFactory{
		Config: LinearModelConfig{
			Data: Dataset{
				X:        mat.NewDense(10, 10, nil),
				Y:        mat.NewVecDense(10, nil),
				ColNames: make([]string, 10),
			},
		},
	}
	ga.Minimize(factory.Generate)
}
