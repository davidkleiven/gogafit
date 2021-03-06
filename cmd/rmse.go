package cmd

import (
	"log"
	"math"

	"github.com/davidkleiven/gogafit/gafit"
	"github.com/spf13/cobra"
)

// rmseCmd represents the rmse command
var rmseCmd = &cobra.Command{
	Use:   "rmse",
	Short: "Calculate RMSE for a model",
	Long: `Calcualte the root mean square error for a given model.

The prediction is given by p = X.dot(c) where X is the design matrix and y is
c is the coefficient vector. RMSE is given by sqrt(mean((p - y)**2)).

The X and y vector is extracted from a csv file with the format (mydata.csv in the example below)

feat1, feat2, feat3
0.1, 0.2, 0.5
-0.4, 0.2, 0.5

the column passed as target is used as the y-vector and the remaining columns are used as the X
matrix.

The coefficient vector is extracted from a csv file of the form (mycoeff.json in the example below)

  {
	"TargetName": "feat3",
	"Datafile": "gafit/_testdata/dataset.csv",
	"Coeffs": {
	  "Var1": 2.9999990000004804,
	  "Var2": 1.0000003999997198
	},
	"Score": {
	  "Name": "aicc",
	  "Value": -25.7622420881808
	}
  }

where the first column is a name (that must match one of header fields in the data csvfile) and
the second column is the value of the coefficients. Coefficients corresponding to columns in the
data matrix that are not listed, is taken as zero.

Minimal example:

gogafit rmse -d mydata.csv -c mycoeff,csv
	`,
	Run: func(cmd *cobra.Command, args []string) {
		dataFile, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		coeffFile, err := cmd.Flags().GetString("model")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		model, err := gafit.ReadModel(coeffFile)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		data, err := gafit.Read(dataFile, model.TargetName)

		if err != nil {
			log.Fatalf("%s\n", err)
		}

		coeffs := model.Coeffs

		pred := data.Dot(coeffs)

		rss := 0.0
		for i := 0; i < pred.Len(); i++ {
			rss += math.Pow(pred.AtVec(i)-data.Y.AtVec(i), 2)
		}
		rmse := math.Sqrt(rss / float64(pred.Len()))
		log.Printf("RMSE: %f\n", rmse)

		// Calculate GCV
		names := []string{}
		for k := range coeffs {
			names = append(names, k)
		}
		X := data.Submatrix(names)
		gcv := gafit.GeneralizedCV(rmse, X)
		log.Printf("Generalized CV (GCV): %f\n", gcv)
	},
}

func init() {
	rootCmd.AddCommand(rmseCmd)

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// rmseCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	rmseCmd.Flags().StringP("data", "d", "", "Csv file with data")
	rmseCmd.Flags().StringP("model", "m", "model.json", "JSON file with fitted model coefficients")
}
