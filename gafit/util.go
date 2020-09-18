package gafit

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const log2pi = 1.83787706641

// CostFunction is a type used to represent cost functions for fitting
type CostFunction func(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64

// Aic returns Afaike's information criteria
func Aic(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	k := float64(coeff.Len())
	logL := LogLikelihood(X, y, coeff)
	return 2.0*k - 2.0*logL
}

// LogLikelihood returns the logarithm of the likelihood function, assuming normal distributed
// variable
func LogLikelihood(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	rss := Rss(X, y, coeff)
	n := float64(y.Len())
	if rss < 1e-10 {
		rss = 1e-10
	}
	return -0.5 * n * (log2pi + 1.0 + math.Log(rss/n))
}

// Aicc returns the corrected Afaike's information criteria
func Aicc(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	k := float64(coeff.Len())
	n := float64(y.Len())

	denum := n - k - 1
	if denum < 1 {
		denum = 1
	}
	return Aic(X, y, coeff) + 2*k*(k+1)/denum
}

// Rss returns the residual sum of square
func Rss(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	pred := Pred(X, coeff)
	rss := 0.0
	for i := 0; i < pred.Len(); i++ {
		rss += math.Pow(pred.AtVec(i)-y.AtVec(i), 2)
	}
	return rss
}

// Rmse returns the residual mean square error
func Rmse(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	n := float64(y.Len())
	return math.Sqrt(Rss(X, y, coeff) / n)
}

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

func allEqualString(s1 []string, s2 []string) bool {
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
