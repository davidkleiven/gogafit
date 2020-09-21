package gafit

import "gonum.org/v1/gonum/mat"

// Dataset is a type that represents a linear model
type Dataset struct {
	X *mat.Dense
	Y *mat.VecDense

	// ColNames gives the name of the "feature" stored in each column of X
	ColNames   []string
	TargetName string
}

// IsEqual returns true if the two dataseta are equal
func (data Dataset) IsEqual(other Dataset) bool {
	tol := 1e-6
	return matrixEqual(data.X, other.X, tol) &&
		vectorEqual(data.Y, other.Y, tol) &&
		allEqualString(data.ColNames, other.ColNames)
}

// NumFeatures return the number of features
func (data Dataset) NumFeatures() int {
	_, c := data.X.Dims()
	return c
}

// NumData returns the number of datapoints
func (data Dataset) NumData() int {
	r, _ := data.X.Dims()
	return r
}

// IncludedFeatures returns the features being included according to the
// passed indicator. 1: feature is included, 0: feature is not included
func (data Dataset) IncludedFeatures(indicator []int) []string {
	names := []string{}
	for i := range indicator {
		if indicator[i] == 1 {
			names = append(names, data.ColNames[i])
		}
	}
	return names
}
