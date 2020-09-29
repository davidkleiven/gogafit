package gafit

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// PredictionErrorFIC tries to select the model that has the highest precision
// for a subset of the data
type PredictionErrorFIC struct {
	Data []int
}

// Evaluate evaluates the focused information criteria
func (pef *PredictionErrorFIC) Evaluate(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	r, c := X.Dims()
	rss := Rss(X, y, coeff)
	cov, err := CovMatrix(X, rss)
	if err != nil {
		return math.Inf(1)
	}

	variancePredError := 0.0
	res := mat.NewDense(1, 1, nil)
	for _, c := range pef.Data {
		res.Product(X.RowView(c).T(), cov, X.RowView(c))
		variancePredError += res.At(0, 0)
	}

	totalVariance := rss/float64(r-c) + variancePredError
	return math.Sqrt(totalVariance)
}
