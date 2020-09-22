package gafit

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

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

// Submatrix returns a submatrix corresponding to columns given
func (data Dataset) Submatrix(names []string) *mat.Dense {
	idxMap := make(map[string]int)
	for i, n := range data.ColNames {
		idxMap[n] = i
	}

	// Check that all names exist
	for _, n := range names {
		if _, ok := idxMap[n]; !ok {
			msg := fmt.Sprintf("Name %s is not a feature in this dataset\n", n)
			panic(msg)
		}
	}

	S := mat.NewDense(data.NumData(), len(names), nil)
	for i, n := range names {
		for j := 0; j < data.NumData(); j++ {
			S.Set(j, i, data.X.At(j, idxMap[n]))
		}
	}
	return S
}

// Dot perform dot product between X and a sparse coefficient vector
// given as a map of strings, where the key is a column name
func (data Dataset) Dot(coeff map[string]float64) *mat.VecDense {
	coeffVec := mat.NewVecDense(data.NumFeatures(), nil)
	for i, key := range data.ColNames {
		if v, ok := coeff[key]; ok {
			coeffVec.SetVec(i, v)
		}
	}
	res := mat.NewVecDense(data.NumData(), nil)
	res.MulVec(data.X, coeffVec)
	return res
}
