package gafit

import (
	"math"
	"math/rand"

	"github.com/MaxHalford/eaopt"
	"gonum.org/v1/gonum/mat"
)

// LinearModelConfig contains static configuration for a linear model
// It contains meta-information needed to fully define a LinearModel
type LinearModelConfig struct {
	Design       *mat.Dense
	Target       *mat.VecDense
	Cost         CostFunction
	MutationRate float64
	NumSplits    int
}

// IsEqual if other is equal to lmc, return true. Otherwise, return false.
func (lmc LinearModelConfig) IsEqual(other LinearModelConfig) bool {
	tol := 1e-6
	return matrixEqual(lmc.Design, other.Design, tol) &&
		vectorEqual(lmc.Target, other.Target, tol) &&
		(math.Abs(lmc.MutationRate-other.MutationRate) < tol) &&
		(lmc.NumSplits == other.NumSplits)
}

// LinearModel represent a genome
type LinearModel struct {
	Config  LinearModelConfig
	include []int
}

// IsEqual returns true of the two models are equal
func (l *LinearModel) IsEqual(other LinearModel) bool {
	return l.Config.IsEqual(other.Config) && AllEqualInt(l.include, other.include)
}

func (l *LinearModel) includedCols() []int {
	cols := []int{}
	for i := range l.include {
		if l.include[i] == 1 {
			cols = append(cols, i)
		}
	}
	return cols
}

// MutationRate returns the mutation rate. If not specified in Config (e.g. 0.0),
// a default value of 0.5 is used
func (l *LinearModel) MutationRate() float64 {
	if l.Config.MutationRate < 1e-16 {
		// Mutation rate not given, use default
		return 0.5
	}
	return l.Config.MutationRate
}
func (l *LinearModel) subMatrix() *mat.Dense {
	rows, _ := l.Config.Design.Dims()
	cols := l.includedCols()

	subMat := mat.NewDense(rows, len(cols), nil)
	for i := 0; i < rows; i++ {
		for j := range cols {
			subMat.Set(i, j, l.Config.Design.At(i, cols[j]))
		}
	}
	return subMat
}

// Evaluate evaluates the fitness
func (l *LinearModel) Evaluate() (float64, error) {
	subMat := l.subMatrix()
	coeff := Fit(subMat, l.Config.Target)
	return l.Config.Cost(l.subMatrix(), l.Config.Target, coeff), nil
}

// Mutate introduces mutations
func (l *LinearModel) Mutate(rng *rand.Rand) {
	for i := range l.include {
		if rng.Float64() < l.Config.MutationRate {
			l.include[i] = (l.include[i] + 1) % 2
		}
	}
}

// Clone create a copy
func (l *LinearModel) Clone() eaopt.Genome {
	model := LinearModel{
		Config:  l.Config,
		include: make([]int, len(l.include)),
	}
	copy(model.include, l.include)
	return &model
}

// Crossover performs a cross over
func (l *LinearModel) Crossover(other eaopt.Genome, rng *rand.Rand) {
	eaopt.CrossGNXInt(l.include, other.(*LinearModel).include, 2, rng)
}
