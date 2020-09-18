package gafit

import (
	"gonum.org/v1/gonum/mat"
)

// CostFunction is a type used to represent cost functions for fitting
type CostFunction func(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64

// Pred predicts the outcome of the linear model
func Pred(X *mat.Dense, coeff *mat.VecDense) *mat.VecDense {
	r, c := X.Dims()

	if c != coeff.Len() {
		panic("Coefficient vector must match the number of features")
	}

	res := mat.NewVecDense(r, nil)
	res.MulVec(X, coeff)
	return res
}

// Fit returns the solution of X*c = y
func Fit(X *mat.Dense, y *mat.VecDense) *mat.VecDense {
	_, c := X.Dims()
	coeff := mat.NewVecDense(c, nil)
	coeff.SolveVec(X, y)
	return coeff
}

// AllEqualInt check if all elements in s1 equals s2
func AllEqualInt(s1 []int, s2 []int) bool {
	if len(s1) != len(s2) {
		return false
	}

	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

func matrixEqual(X *mat.Dense, Y *mat.Dense, tol float64) bool {
	if (X == nil) && (Y == nil) {
		return true
	}

	if (X == nil) || (Y == nil) {
		return false
	}
	return mat.EqualApprox(X, Y, tol)
}

func vectorEqual(X *mat.VecDense, Y *mat.VecDense, tol float64) bool {
	if (X == nil) && (Y == nil) {
		return true
	}

	if (X == nil) || (Y == nil) {
		return false
	}
	return mat.EqualApprox(X, Y, tol)
}
