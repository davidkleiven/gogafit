package gafit

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func completeModel() LinearModel {
	return LinearModel{
		Config: LinearModelConfig{
			Data: Dataset{
				X: mat.NewDense(2, 2, []float64{1.0, 2.0, 1.0, -1.0}),
				Y: mat.NewVecDense(2, []float64{4.0, -3.0}),
			},
			Cost: func(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
				return 0.0
			},
			MutationRate: 0.3,
			NumSplits:    1,
		},
		Include: []int{1, 1},
	}
}

func rng() *rand.Rand {
	src := rand.NewSource(42)
	return rand.New(src)
}

func TestMutationRate(t *testing.T) {
	model := LinearModel{}

	tol := 1e-16
	if math.Abs(model.MutationRate()-0.5) > tol {
		t.Errorf("Expected default value 0.5 got %f\n", model.MutationRate())
	}

	model.Config.MutationRate = 0.3
	if math.Abs(model.MutationRate()-0.3) > tol {
		t.Errorf("Expected default value 0.3 got %f\n", model.MutationRate())
	}
}

func TestSubMatrix(t *testing.T) {
	model := LinearModel{
		Config: LinearModelConfig{
			Data: Dataset{
				X: mat.NewDense(2, 7, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
					8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0}),
			},
		},
		Include: []int{0, 0, 1, 0, 1, 0, 1},
	}

	want := mat.NewDense(2, 3, []float64{3.0, 5.0, 7.0, 10.0, 12.0, 14.0})
	got := model.subMatrix()

	tol := 1e-6
	if !mat.EqualApprox(want, got, tol) {
		t.Errorf("Expected\n%v\ngot\n%v\n", mat.Formatted(want), mat.Formatted(got))
	}
}

func TestEqual(t *testing.T) {
	for i, test := range []struct {
		Mod1   LinearModel
		Mod2   LinearModel
		Expect bool
	}{
		// Empty models
		{
			Mod1:   LinearModel{},
			Mod2:   LinearModel{},
			Expect: true,
		},
		{
			Mod1: LinearModel{
				Config: LinearModelConfig{
					Data: Dataset{
						X: mat.NewDense(2, 1, []float64{2.0, 3.0}),
					},
				},
			},
			Mod2:   LinearModel{},
			Expect: false,
		},
	} {
		got := test.Mod1.IsEqual(test.Mod2)
		if got != test.Expect {
			t.Errorf("Test #%d: Expected %v got %v\n", i, test.Expect, got)
		}
	}
}

func TestClone(t *testing.T) {
	model := completeModel()
	mod2 := model.Clone().(*LinearModel)
	if !model.IsEqual(*mod2) {
		t.Errorf("Clone produces a model that is different from the parent.")
	}
}

func TestOptimize(t *testing.T) {
	model := completeModel()

	// Confirm that model is unchagned after optimize
	origMod := model.Clone().(*LinearModel)
	res1 := model.Optimize()

	if !model.IsEqual(*origMod) {
		t.Errorf("Model changes after call to optimize")
	}

	res2 := model.Optimize()

	if !res1.IsEqual(res2) {
		t.Errorf("Two successive calls yields different results")
	}
}

func TestMutate(t *testing.T) {
	model := completeModel()
	original := make([]int, len(model.Include))
	copy(original, model.Include)

	model.Config.MutationRate = 0.99
	model.Mutate(rng())

	if AllEqualInt(original, model.Include) {
		t.Errorf("Bits were not mutated")
	}
}

func TestCrossOver(t *testing.T) {
	model := completeModel()
	model2 := completeModel()

	// artificially make the Include arrays longer
	model.Include = make([]int, 100)
	model2.Include = make([]int, 100)
	r := rng()
	for i := 0; i < 100; i++ {
		if r.Float64() < 0.5 {
			model.Include[i] = 1
		} else {
			model.Include[i] = 0
		}

		if r.Float64() < 0.5 {
			model2.Include[i] = 1
		} else {
			model2.Include[i] = 0
		}
	}
	model2.Include[0] = 0

	orig := make([]int, len(model.Include))
	copy(orig, model.Include)

	orig2 := make([]int, len(model2.Include))
	copy(orig2, model2.Include)

	model.Crossover(&model2, r)

	if AllEqualInt(orig, model.Include) || AllEqualInt(orig2, model2.Include) {
		t.Errorf("Crossover did not alter model")
	}
}

func TestEvaluate(t *testing.T) {
	model := completeModel()

	want := 0.0
	got, err := model.Evaluate()

	if err != nil {
		t.Errorf("Evaluate returned error\n")
	}

	if math.Abs(want-got) > 1e-6 {
		t.Errorf("Expected %f got %f\n", want, got)
	}
}
