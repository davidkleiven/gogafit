package gafit

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/MaxHalford/eaopt"
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

	// All models with a higher coefficient of determination than this, is
	// considered equally good. This is added to the rmss to avoid log 0 for
	// perfect predictions
	targetRsq := 0.999999
	tol := math.Sqrt((1.0-targetRsq)*meanSumOfSquares(y)) + 1e-10
	return -0.5 * n * (log2pi + 1.0 + math.Log(rmss+tol))
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

func numericRange(x *mat.VecDense) (float64, float64) {
	if x.Len() == 0 {
		return 0.0, 0.0
	}

	minval := x.AtVec(0)
	maxval := x.AtVec(0)

	for i := 0; i < x.Len(); i++ {
		if x.AtVec(i) < minval {
			minval = x.AtVec(i)
		}

		if x.AtVec(i) > maxval {
			maxval = x.AtVec(i)
		}
	}
	return minval, maxval
}

func mean(y *mat.VecDense) float64 {
	if y.Len() == 0 {
		return 0.0
	}

	sum := 0.0
	for i := 0; i < y.Len(); i++ {
		sum += y.AtVec(i)
	}
	return sum / float64(y.Len())
}

func meanSumOfSquares(y *mat.VecDense) float64 {
	yMean := mean(y)
	sst := 0.0
	for i := 0; i < y.Len(); i++ {
		sst += math.Pow(y.AtVec(i)-yMean, 2)
	}
	return sst / float64(y.Len())
}

// subMatrix extracts a submatrix from X using only the columns specified
func subMatrix(X *mat.Dense, cols []int) *mat.Dense {
	rows, _ := X.Dims()
	res := mat.NewDense(rows, len(cols), nil)
	for i, c := range cols {
		for j := 0; j < rows; j++ {
			res.Set(j, i, X.At(j, c))
		}
	}
	return res
}

// CovMatrix calculates the covariance matrix between the coefficients
func CovMatrix(X *mat.Dense, rss float64) (*mat.SymDense, error) {
	r, c := X.Dims()
	if c > r {
		return nil, errors.New("CovMatrix used QR factorization. The system must be underdetermined")
	}

	var qr mat.QR
	qr.Factorize(X)

	var R mat.Dense
	qr.RTo(&R)

	var cholesky mat.Cholesky
	_, n := R.Dims()
	Rtri := mat.NewTriDense(n, mat.Upper, R.RawMatrix().Data[:n*n])
	cholesky.SetFromU(Rtri)

	res := mat.NewSymDense(n, nil)
	err := cholesky.InverseTo(res)
	if err != nil {
		return nil, err
	}

	scale := 1.0
	if r > c {
		scale = 1.0 / float64(r-c)
	}

	res.ScaleSym(scale*rss, res)
	return res, nil
}

// ReadModel reads a model from a JSON file
func ReadModel(fname string) (Model, error) {
	jsonFile, err := os.Open(fname)
	if err != nil {
		return Model{}, nil
	}
	defer jsonFile.Close()
	bytes, err := ioutil.ReadAll(jsonFile)

	if err != nil {
		return Model{}, err
	}

	var model Model
	json.Unmarshal(bytes, &model)
	return model, nil
}

// Score is a conveniene type used to collect information about the quality of a model
type Score struct {
	Name  string
	Value float64
}

// Model is convenience type used to store information about a model
type Model struct {
	Datafile string
	Coeffs   map[string]float64
	Score    Score
}

// NewModel creates a new fitted model from the best individual of a GA run
func NewModel(best eaopt.Individual, dataset Dataset, cost string, datafile string) Model {
	bestMod := best.Genome.(*LinearModel)
	res := bestMod.Optimize()
	coeff := res.Coeff.RawVector().Data
	features := dataset.IncludedFeatures(res.Include)
	model := Model{
		Datafile: datafile,
		Score: Score{
			Name:  cost,
			Value: res.Score,
		},
		Coeffs: join2map(features, coeff),
	}
	return model
}

func join2map(keys []string, values []float64) map[string]float64 {
	res := make(map[string]float64)
	for i := range keys {
		res[keys[i]] = values[i]
	}
	return res
}

// SaveModel writes a JSON version of the model to file
func SaveModel(fname string, model Model) error {
	modelSerialized, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(fname, modelSerialized, 0644)
	return err
}

// GABackupCB is a default type used to construct a default backup function
type GABackupCB struct {
	Cost       string
	Dataset    Dataset
	DataFile   string
	Rate       uint
	BackupFile string
}

// Build constructs the callback function
func (gab *GABackupCB) Build() func(ga *eaopt.GA) {
	return func(ga *eaopt.GA) {
		if ga.Generations%gab.Rate == 0 {
			log.Printf("Best %s at generation %d: %f\n", gab.Cost, ga.Generations, ga.HallOfFame[0].Fitness)
			model := NewModel(ga.HallOfFame[0], gab.Dataset, gab.Cost, gab.DataFile)
			SaveModel(gab.BackupFile, model)
		}
	}
}

// Prediction is a type that represent a prediction (the expected valud and the standard deviation)
type Prediction struct {
	Value float64
	Std   float64
}

// IsEqual returns true of the two predictions are equal
func (p Prediction) IsEqual(other Prediction) bool {
	tol := 1e-8
	return (math.Abs(p.Value-other.Value)) < tol && (math.Abs(p.Std-other.Std) < tol)
}

// SavePredictions stores the predictions in a file
func SavePredictions(fname string, pred []Prediction) error {
	ofile, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer ofile.Close()

	writer := csv.NewWriter(ofile)
	defer writer.Flush()
	err = writer.Write([]string{"prediction, stddev"})
	if err != nil {
		return err
	}
	for _, p := range pred {
		record := []string{fmt.Sprintf("%f", p.Value), fmt.Sprintf("%f", p.Std)}
		err = writer.Write(record)
		if err != nil {
			return err
		}
	}
	return nil
}

// ReadPredictions reads the predictions from a csv file (same as stored by SavePredictions)
func ReadPredictions(fname string) ([]Prediction, error) {
	infile, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer infile.Close()

	reader := csv.NewReader(infile)
	pred := []Prediction{}
	lineNo := 0
	for {
		line, err := reader.Read()
		lineNo++

		// First line a header
		if lineNo == 1 {
			continue
		}
		if err == io.EOF {
			return pred, nil
		}
		value, err := strconv.ParseFloat(line[0], 64)
		if err != nil {
			return pred, err
		}
		std, err := strconv.ParseFloat(line[1], 64)
		if err != nil {
			return pred, err
		}

		pred = append(pred, Prediction{Value: value, Std: std})
	}

}

// GetPredictions together with the standard deviations for all data in predData. If predData
// is nil, data will be used (e.g. in sample prediction errors)
func GetPredictions(data Dataset, model Model, predData *Dataset) []Prediction {
	names := []string{}
	coeffs := mat.NewVecDense(len(model.Coeffs), nil)
	counter := 0

	for k, v := range model.Coeffs {
		names = append(names, k)
		coeffs.SetVec(counter, v)
		counter++
	}

	sub := data.Submatrix(names)

	if predData == nil {
		predData = &data
	}

	rss := Rss(sub, data.Y, coeffs)
	pred := Pred(sub, coeffs)
	numData, numFeat := sub.Dims()
	denum := numData - numFeat
	if denum <= 0 {
		denum = 1
	}
	correctedRss := rss / float64(denum)
	cov, err := CovMatrix(sub, rss)
	if err != nil {
		panic(err)
	}

	subPred := predData.Submatrix(names)
	r, _ := subPred.Dims()
	variance := mat.NewDense(r, r, nil)
	variance.Product(subPred, cov, subPred.T())

	predictions := make([]Prediction, pred.Len())
	for i := 0; i < pred.Len(); i++ {
		predictions[i].Value = pred.AtVec(i)
		predictions[i].Std = math.Sqrt(correctedRss + variance.At(i, i))
	}
	return predictions
}
