package gafit

import (
	"math"
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCaptureCostFuncValue(t *testing.T) {
	key := "cost"
	tol := 1e-6
	for i, test := range []struct {
		out    string
		expect float64
	}{
		{
			out:    CostFunctionIdentifier + " 1.02",
			expect: 1.02,
		},
		{
			out:    CostFunctionIdentifier + " 0.0002",
			expect: 0.0002,
		},
		{
			out:    CostFunctionIdentifier + " 0.020",
			expect: 0.02,
		},
		{
			out:    CostFunctionIdentifier + " 10.2",
			expect: 10.2,
		},
		{
			out:    "My program\n" + CostFunctionIdentifier + " 10.2\nOther Info. Finished\n",
			expect: 10.2,
		},
	} {
		res, err := captureCostFuncValue(test.out)
		if err != nil {
			t.Errorf("Test #%d: Error %s\n", i, err)
		}
		got := res.GetFloat(key)

		if math.Abs(got-test.expect) > tol {
			t.Errorf("Test #%d: Expected %f got %f\n", i, got, test.expect)
		}

	}
}

func TestCostFuncHook(t *testing.T) {
	script, err := DemoCostFuncPython("python")
	if err != nil {
		t.Errorf("Error %s\n", err)
		return
	}
	outfile := "./myhook.py"

	f, err := os.OpenFile(outfile, os.O_CREATE|os.O_WRONLY, 0755)
	if err != nil {
		t.Errorf("%s\n", err)
		return
	}

	if _, err = f.WriteString(script); err != nil {
		t.Errorf("%s\n", err)
		f.Close()
		return
	}
	f.Close()

	X := mat.NewDense(3, 3, nil)
	Y := mat.NewVecDense(3, nil)
	coeff := mat.NewVecDense(3, nil)

	hook := NewCostFunctionHook(outfile)
	res := hook.Execute(X, Y, coeff)

	tol := 1e-6
	if math.Abs(res-0.6) > tol {
		t.Errorf("Expected 0.6 got %f\n", res)
	}

	err = os.Remove(outfile)
}

func TestModel2JSON(t *testing.T) {
	X := mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	Y := mat.NewVecDense(2, []float64{1.0, 2.0})
	coeff := mat.NewVecDense(2, []float64{2.0, 3.0})
	res := model2json(X, Y, coeff)

	expect := "{\"Rows\":2,\"Cols\":3,\"X\":[1,2,3,4,5,6],\"Y\":[1,2],\"Coeff\":[2,3]}"

	if expect != res {
		t.Errorf("Expected\n%s\ngot\n%s\n", expect, res)
	}
}
