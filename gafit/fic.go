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
	rss := Rss(X, y, coeff)
	pred := Pred(X, coeff)
	cov, err := CovMatrix(X, rss)
	if err != nil {
		return math.Inf(1)
	}

	variancePredError := 0.0
	biasSq := 0.0
	res := mat.NewDense(1, 1, nil)
	for _, c := range pef.Data {
		res.Product(X.RowView(c).T(), cov, X.RowView(c))
		variancePredError += res.At(0, 0)
		biasSq += math.Pow(y.AtVec(c)-pred.AtVec(c), 2.0)
	}

	totalVariance := biasSq + variancePredError
	return math.Sqrt(totalVariance)
}
