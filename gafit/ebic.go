package gafit

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/combin"
)

// EBic is a type used to calculate the extended BIC criterion.
// An implicit underlying assumption for BIC is that the prior distribution
// is constant for all models. This may not be feasible when the number of
// features are large. EBIC tries to penalize large models higher than BIC,
// by setting the prior distribution inversely proportional to the total
// number of models with a given size.
// If we have N features, and k featurea are selected then the prior p(s)
// is proportional to tau^{gamma}, where tau is the total number of models
// with that size (e.g. tau = N!/(k!(N-k)!)) and 0 <= gamma <= 1 is a tuning
// constnat. If gamma is zero, then EBIC is equal to BIC
type EBic struct {
	Gamma          float64
	MaxNumFeatures int
}

// Evaluate evaluates the EBic criterion
func (e EBic) Evaluate(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense) float64 {
	numFeat := float64(coeff.Len())
	return Bic(X, y, coeff) + 2.0*e.Gamma*combin.LogGeneralizedBinomial(float64(e.MaxNumFeatures), numFeat)
}

// NewDefaultEBic returns a new Ebic function
func NewDefaultEBic(maxNumFeat int) EBic {
	return EBic{
		Gamma:          1.0,
		MaxNumFeatures: maxNumFeat,
	}
}
