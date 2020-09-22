/*
Copyright © 2020 NAME HERE <EMAIL ADDRESS>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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

The coefficient vector is extracted from a csv file of the form (mycoeff.csv in the example below)

feat1, -0.5
feat2, 0.6

where the first column is a name (that must match one of header fields in the data csvfile) and
the second column is the value of the coefficients. Coefficients corresponding to columns in the
data matrix that are not listed, is taken as zero.

Minimal example:

gogafit rmse -d mydata.csv -t feat3 -c mycoeff,csv
	`,
	Run: func(cmd *cobra.Command, args []string) {
		dataFile, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		target, err := cmd.Flags().GetString("target")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		target, err = ClosestHeaderName(dataFile, target)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
		log.Printf("Using %s as target column\n", target)

		coeffFile, err := cmd.Flags().GetString("coeff")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		data, err := gafit.Read(dataFile, target)

		if err != nil {
			log.Fatalf("%s\n", err)
		}

		coeffs, err := ReadCoeffs(coeffFile)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

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
	rmseCmd.Flags().StringP("target", "y", "", "Name of target column")
	rmseCmd.Flags().StringP("coeff", "c", "model.csv", "CSV file with fitted model coefficients")
}
