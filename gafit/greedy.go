package gafit

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// OrthogonalMatchingPursuit optimizes the cost function by selecting the model that leads to the
// largest decrease in the cost function
func OrthogonalMatchingPursuit(dataset Dataset, cost CostFunction, maxFeatures int) OptimizeResult {
	X := dataset.X
	y := dataset.Y
	Xnorm := mat.DenseCopyOf(X)
	normalize(Xnorm)
	_, cols := Xnorm.Dims()
	residuals := mat.VecDenseCopyOf(y)
	proj := mat.NewVecDense(cols, nil)

	selected := []int{}
	names := []string{}
	bestScore := math.Inf(1)
	bestSelection := make([]int, 0, cols)
	end := maxFeatures
	if cols < end {
		end = cols
	}

	for i := 0; i < end; i++ {
		proj.MulVec(Xnorm.T(), residuals)
		best := argAbsMax(proj)
		selected = append(selected, best)
		names = append(names, dataset.ColNames[best])
		sub := subMatrix(Xnorm, selected)
		tempCoeff := Fit(sub, y)

		score := cost(sub, y, tempCoeff, names)

		if score < bestScore {
			bestScore = score
			bestSelection = bestSelection[:0]
			for _, v := range selected {
				bestSelection = append(bestSelection, v)
			}
		}
		pred := Pred(sub, tempCoeff)
		residuals.SubVec(y, pred)
	}

	// Perform a fit with the unnormalized matrix
	sort.Ints(bestSelection)
	sub := subMatrix(X, bestSelection)
	coeff := Fit(sub, y)

	// Convert selection to an include-bit string
	return OptimizeResult{
		Score:   bestScore,
		Coeff:   coeff,
		Include: selection2bitstring(bestSelection, cols),
	}
}

func normalize(X *mat.Dense) {
	rows, cols := X.Dims()
	tol := 1e-16
	for i := 0; i < cols; i++ {
		length := math.Sqrt(mat.Dot(X.ColView(i), X.ColView(i)))
		if length > tol {
			for j := 0; j < rows; j++ {
				X.Set(j, i, X.At(j, i)/length)
			}
		}
	}
}

func allConstant(v mat.Vector, tol float64) bool {
	for i := 0; i < v.Len(); i++ {
		if math.Abs(v.AtVec(i)-v.AtVec(0)) > tol {
			return false
		}
	}
	return true
}

func standardize(X *mat.Dense) {
	rows, cols := X.Dims()

	// Subtract mean
	for i := 0; i < cols; i++ {
		if !allConstant(X.ColView(i), 1e-6) {
			mean := mat.Sum(X.ColView(i)) / float64(rows)
			for j := 0; j < rows; j++ {
				X.Set(j, i, X.At(j, i)-mean)
			}
		}
	}
	normalize(X)
}

// argAbsMax returns the index of the element that has the maximum absolute value
func argAbsMax(v *mat.VecDense) int {
	max := math.Abs(v.AtVec(0))
	maxIdx := 0
	for i := 0; i < v.Len(); i++ {
		if math.Abs(v.AtVec(i)) > max {
			max = math.Abs(v.AtVec(i))
			maxIdx = i
		}
	}
	return maxIdx
}

func selection2bitstring(selected []int, length int) []int {
	include := make([]int, length)
	for _, s := range selected {
		include[s] = 1
	}
	return include
}

func subtractMean(v *mat.VecDense) {
	mean := mat.Sum(v) / float64(v.Len())
	for i := 0; i < v.Len(); i++ {
		v.SetVec(i, v.AtVec(i)-mean)
	}
}
