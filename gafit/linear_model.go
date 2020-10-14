package gafit

import (
	"log"
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

	// MaxFeatToDataRatio specifies the maximum value of #feat/#data. If not given,
	// a default value of 0.5 is used
	MaxFeatToDataRatio float64
}

// GetCostFunction returns the cost function. If not given, AICC is used as default
func (lmc *LinearModelConfig) GetCostFunction() CostFunction {
	if lmc.Cost == nil {
		log.Printf("No cost function set. Using AICC as default.\n")
		lmc.Cost = Aicc
	}
	return lmc.Cost
}

// IsEqual if other is equal to lmc, return true. Otherwise, return false.
func (lmc LinearModelConfig) IsEqual(other LinearModelConfig) bool {
	tol := 1e-6
	return lmc.Data.IsEqual(other.Data) &&
		(math.Abs(lmc.MutationRate-other.MutationRate) < tol) &&
		(lmc.NumSplits == other.NumSplits)
}

func (lmc LinearModelConfig) getMaxFeatToDataRatio() float64 {
	if lmc.MaxFeatToDataRatio < 1e-10 {
		return 0.5
	}
	return lmc.MaxFeatToDataRatio
}

// LargestModel returns the largest model consistent with the feature to data ratio
func (lmc LinearModelConfig) LargestModel() int {
	return int(lmc.getMaxFeatToDataRatio() * float64(lmc.Data.NumData()))
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
func (l *LinearModel) subDataset() Dataset {
	rows, _ := l.Config.Data.X.Dims()
	cols := l.IncludedCols()

	data := Dataset{
		X:          mat.NewDense(rows, len(cols), nil),
		Y:          l.Config.Data.Y,
		ColNames:   make([]string, len(cols)),
		TargetName: l.Config.Data.TargetName,
	}

	for i := 0; i < rows; i++ {
		for j := range cols {
			data.X.Set(i, j, l.Config.Data.X.At(i, cols[j]))
		}
	}

	for count, col := range cols {
		data.ColNames[count] = l.Config.Data.ColNames[col]
	}
	return data
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
	subMat := l.subDataset().X
	return Fit(subMat, l.Config.Data.Y)
}

// Evaluate evaluates the fitness
func (l *LinearModel) Evaluate() (float64, error) {
	if l.IsEmpty() {
		panic("The model is empty.")
	}
	return l.Optimize().Score, nil
}

// NumIncluded returns the number of included columns
func (l *LinearModel) NumIncluded() int {
	return len(l.IncludedCols())
}

// Optimize flips all inclusions in. After a call to this
// function, the included features are affected and set to the best genome
func (l *LinearModel) Optimize() OptimizeResult {
	data := l.subDataset()
	greedyRes := OrthogonalMatchingPursuit(data, l.Config.GetCostFunction(), l.Config.LargestModel())
	res := OptimizeResult{
		Score:   greedyRes.Score,
		Coeff:   greedyRes.Coeff,
		Include: make([]int, len(l.Include)),
	}
	cols := l.IncludedCols()
	for i, v := range greedyRes.Include {
		if v == 1 {
			res.Include[cols[i]] = 1
		}
	}
	return res
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
	mutType := rng.Int31n(2)

	switch mutType {
	case 0:
		// Flip random bits
		flipMutation(l.Include, rng, l.Config.MutationRate)
		break
	case 1:
		// Sparsify mutation, remove 50% of the active values
		sparsifyMutation(l.Include, rng, 0.5)
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

func flipMutation(array []int, rng *rand.Rand, threshold float64) {
	for i := range array {
		if rng.Float64() < threshold {
			array[i] = (array[i] + 1) % 2
		}
	}
}

func sparsifyMutation(array []int, rng *rand.Rand, frac float64) {
	nonzero := make([]int, len(array))
	num := 0
	for i := range array {
		if array[i] == 1 {
			nonzero[num] = i
			num++
		}
	}

	nonzero = nonzero[:num]
	rng.Shuffle(len(nonzero), func(i, j int) { nonzero[i], nonzero[j] = nonzero[j], nonzero[i] })
	numFlip := int(frac * float64(len(nonzero)))
	for _, idx := range nonzero[:numFlip] {
		array[idx] = 0
	}
}
