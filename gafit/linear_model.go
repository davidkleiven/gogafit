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
	Data         Dataset
	Cost         CostFunction
	MutationRate float64
	NumSplits    uint
}

// IsEqual if other is equal to lmc, return true. Otherwise, return false.
func (lmc LinearModelConfig) IsEqual(other LinearModelConfig) bool {
	tol := 1e-6
	return lmc.Data.IsEqual(other.Data) &&
		(math.Abs(lmc.MutationRate-other.MutationRate) < tol) &&
		(lmc.NumSplits == other.NumSplits)
}

// LinearModel represent a genome
type LinearModel struct {
	Config  LinearModelConfig
	Include []int
}

// IsEqual returns true of the two models are equal
func (l *LinearModel) IsEqual(other LinearModel) bool {
	return l.Config.IsEqual(other.Config) && AllEqualInt(l.Include, other.Include)
}

// IncludedCols return the index of the columns that are included according to the
// 1/0 values in inclue (1: included, 0: excluded)
func (l *LinearModel) IncludedCols() []int {
	cols := []int{}
	for i := range l.Include {
		if l.Include[i] == 1 {
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
	rows, _ := l.Config.Data.X.Dims()
	cols := l.IncludedCols()

	subMat := mat.NewDense(rows, len(cols), nil)
	for i := 0; i < rows; i++ {
		for j := range cols {
			subMat.Set(i, j, l.Config.Data.X.At(i, cols[j]))
		}
	}
	return subMat
}

// IsEmpty returns true if the model contains no features
func (l *LinearModel) IsEmpty() bool {
	for i := range l.Include {
		if l.Include[i] != 0 {
			return false
		}
	}
	return true
}

func (l *LinearModel) flipRandomIfEmpty(rng *rand.Rand) {
	if l.IsEmpty() {
		l.Include[rng.Int31n(int32(len(l.Include)))] = 1
	}

}

// GetCoeff return the coefficients corresponding to the current selection
func (l *LinearModel) GetCoeff() *mat.VecDense {
	subMat := l.subMatrix()
	return Fit(subMat, l.Config.Data.Y)
}

// Evaluate evaluates the fitness
func (l *LinearModel) Evaluate() (float64, error) {
	if l.IsEmpty() {
		panic("The model is empty.")
	}
	return l.Optimize().Score, nil
}

// Optimize flips all inclusions in. After a call to this
// function, the included features are affected and set to the best genome
func (l *LinearModel) Optimize() OptimizeResult {
	bestInclude := make([]int, len(l.Include))
	copy(bestInclude, l.Include)
	bestCoeff := l.GetCoeff()
	bestScore := l.Config.Cost(l.subMatrix(), l.Config.Data.Y, bestCoeff)
	origInclude := make([]int, len(l.Include))
	copy(origInclude, bestInclude)

	for i := range l.Include {
		old := l.Include[i]
		l.Include[i] = (old + 1) % 2

		if !l.IsEmpty() {
			coeff := l.GetCoeff()
			score := l.Config.Cost(l.subMatrix(), l.Config.Data.Y, coeff)
			if score < bestScore {
				copy(bestInclude, l.Include)
				bestScore = score
				bestCoeff.Reset()
				bestCoeff.CloneFromVec(coeff)
			} else {
				l.Include[i] = old
			}
		} else {
			l.Include[i] = old
		}
	}

	// Leave the model unchanged
	copy(l.Include, origInclude)
	return OptimizeResult{
		Score:   bestScore,
		Coeff:   bestCoeff,
		Include: bestInclude,
	}
}

// OptimizeResult is returned by local optimization of the linear model
type OptimizeResult struct {
	Score   float64
	Include []int
	Coeff   *mat.VecDense
}

// IsEqual returns ture if the two optimize results are equal
func (or *OptimizeResult) IsEqual(other OptimizeResult) bool {
	tol := 1e-6
	return (math.Abs(or.Score-other.Score) < tol) &&
		AllEqualInt(or.Include, other.Include) &&
		mat.EqualApprox(or.Coeff, other.Coeff, tol)
}

// Mutate introduces mutations
func (l *LinearModel) Mutate(rng *rand.Rand) {
	for i := range l.Include {
		if rng.Float64() < l.Config.MutationRate {
			l.Include[i] = (l.Include[i] + 1) % 2
		}
	}
	l.flipRandomIfEmpty(rng)
}

// Clone create a copy
func (l *LinearModel) Clone() eaopt.Genome {
	model := LinearModel{
		Config:  l.Config,
		Include: make([]int, len(l.Include)),
	}
	copy(model.Include, l.Include)
	return &model
}

// NumSplits returns the number of splits used in cross over. If not, set
// 2 is used as default
func (l *LinearModel) NumSplits() uint {
	if l.Config.NumSplits == 0 {
		return 2
	}
	return l.Config.NumSplits
}

// Crossover performs a cross over
func (l *LinearModel) Crossover(other eaopt.Genome, rng *rand.Rand) {
	eaopt.CrossGNXInt(l.Include, other.(*LinearModel).Include, l.NumSplits(), rng)

	// Make sure none og genomes are empty
	l.flipRandomIfEmpty(rng)
	other.(*LinearModel).flipRandomIfEmpty(rng)
}

// LinearModelFactory produces random models
type LinearModelFactory struct {
	Config LinearModelConfig

	// Probability of initialition each features. If not, set default value of 0.5
	// is used. Example: a value of 0.2 will lead to 20% of all features being included
	// in the initial pool
	Prob float64
}

func (lmf *LinearModelFactory) probability() float64 {
	if lmf.Prob < 1e-16 {
		return 0.5
	}
	return lmf.Prob
}

// Generate creates a new random linear model
func (lmf *LinearModelFactory) Generate(rng *rand.Rand) eaopt.Genome {
	model := LinearModel{
		Config:  lmf.Config,
		Include: make([]int, lmf.Config.Data.NumFeatures()),
	}

	for i := range model.Include {
		if rng.Float64() < lmf.probability() {
			model.Include[i] = 1
		} else {
			model.Include[i] = 0
		}
	}
	model.flipRandomIfEmpty(rng)
	return &model
}
