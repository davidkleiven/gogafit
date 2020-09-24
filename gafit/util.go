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
	n := float64(y.Len())
	rmss := Rss(X, y, coeff) / n
	if rmss < 1e-10 {
		rmss = 1e-10
	}
	return -0.5 * n * (log2pi + 1.0 + math.Log(rmss))
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

// Bic returns the Bayes information criterion
func Bic(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	logL := LogLikelihood(X, y, coeff)
	k := float64(coeff.Len())
	n := float64(y.Len())
	return k*math.Log(n) - 2.0*logL
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

// GeneralizedCV returns the generalized CV, given by
// rmse/(1 - Tr(H)/N), where H is the HatMatrix and
// N is the number of datapoints
func GeneralizedCV(rmse float64, X *mat.Dense) float64 {
	H := HatMatrix(X)
	tr := mat.Trace(H)
	N, _ := X.Dims()
	return rmse / (1.0 - tr/float64(N))
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

// FitSVD returns the solution of X*c = y
func FitSVD(X *mat.Dense, y *mat.VecDense) *mat.VecDense {
	_, c := X.Dims()
	var svd mat.SVD
	svd.Factorize(X, mat.SVDThin)

	s := svd.Values(nil)
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)

	var uTdoty mat.VecDense
	uTdoty.MulVec(u.T(), y)

	lamb := 1e-8
	for i := 0; i < len(s); i++ {
		invSigma := s[i] / (s[i]*s[i] + lamb)
		uTdoty.SetVec(i, uTdoty.At(i, 0)*invSigma)
	}
	coeff := mat.NewVecDense(c, nil)
	coeff.MulVec(&v, &uTdoty)
	return coeff
}

// Fit solves the least square problem
func Fit(X *mat.Dense, y *mat.VecDense) *mat.VecDense {
	_, n := X.Dims()
	coeff := mat.NewVecDense(n, nil)
	if err := coeff.SolveVec(X, y); err != nil {
		return FitSVD(X, y)
	}
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

// SubMatrix creates a sub-view of a matrix. The view contains the upper
// left corner starting from element (0, 0) and ending at (Rows, Cols)
type SubMatrix struct {
	X    mat.Matrix
	Rows int
	Cols int
}

// Dims returns the dimansion of the matrix
func (s *SubMatrix) Dims() (int, int) {
	return s.Rows, s.Cols
}

// At returns the value of element (i, j)
func (s *SubMatrix) At(i, j int) float64 {
	return s.X.At(i, j)
}

// T returns the transpose of the matrix
func (s *SubMatrix) T() mat.Matrix {
	return &SubMatrix{
		X:    s.X.T(),
		Rows: s.Cols,
		Cols: s.Rows,
	}
}

// HatMatrix returns the matrix that maps training data onto predictions.
// y = Hy', where y' are training points. In case of linear regression,
// y = Xc, where c is a coefficient vector that is given by c = (X^TX)^{-1}X^Ty',
// the hat matrix H = X(X^TX)^{-1}X^T. Internally, H is calculated by using the QR
// decomposition of R
func HatMatrix(X *mat.Dense) *mat.Dense {
	r, c := X.Dims()

	// If ther number of columns is larger than the number of rows,
	// H maps y' exactly to y.
	if c > r {
		H := mat.NewDense(r, r, nil)
		for i := 0; i < r; i++ {
			H.Set(i, i, 1.0)
		}
		return H
	}

	qr := mat.QR{}
	qr.Factorize(X)

	var Q mat.Dense
	qr.QTo(&Q)

	n, _ := Q.Dims()
	Q1 := SubMatrix{
		X:    &Q,
		Rows: n,
		Cols: c,
	}

	H := mat.NewDense(n, n, nil)
	H.Mul(&Q1, Q1.T())
	return H
}
