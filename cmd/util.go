package cmd

import (
	"encoding/csv"
	"errors"
	"fmt"
	"image/color"
	"io"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/plot/vg/draw"
)

// ClosestHeaderName returns a header name containing <name>
func ClosestHeaderName(fname string, name string) (string, error) {
	f, err := os.Open(fname)
	if err != nil {
		return "", err
	}
	defer f.Close()
	reader := csv.NewReader(f)
	header, err := reader.Read()
	if err != nil {
		return "", err
	}

	for i := range header {
		str := strings.TrimSpace(header[i])
		str = strings.Trim(str, "#/\n")

		if strings.Contains(str, name) {
			return header[i], nil
		}
	}
	msg := fmt.Sprintf("No header contains %s\n", name)
	return "", errors.New(msg)
}

// ReadCoeffs return a map with the coefficients
func ReadCoeffs(fname string) (map[string]float64, error) {
	f, err := os.Open(fname)
	coeff := make(map[string]float64)
	if err != nil {
		return coeff, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	for {
		line, err := reader.Read()
		if err == io.EOF {
			return coeff, nil
		}
		num, err := strconv.ParseFloat(line[1], 64)
		if err != nil {
			return coeff, err
		}
		coeff[line[0]] = num
	}
}

// ColorCycle is a type that represents a color cycle
type ColorCycle struct {
	Colors  []color.RGBA
	Current int
}

// Next returns the color and updates the scheme to the next
func (cc *ColorCycle) Next() color.RGBA {
	c := cc.Colors[cc.Current%len(cc.Colors)]
	cc.Current++
	return c
}

// Get given color
func (cc *ColorCycle) Get(i int) color.RGBA {
	return cc.Colors[i%len(cc.Colors)]
}

// JosephAndHisBrothers return a color cycle extracted by Google arts for the image
// Joseph And His Brothers by
func JosephAndHisBrothers() ColorCycle {
	return ColorCycle{
		Colors: []color.RGBA{
			{
				R: 127,
				G: 44,
				B: 24,
				A: 255,
			},
			{
				R: 69,
				G: 50,
				B: 34,
				A: 255,
			},
			{
				R: 26,
				G: 28,
				B: 20,
				A: 255,
			}, {
				R: 137,
				G: 98,
				B: 57,
				A: 255,
			}, {
				R: 206,
				G: 195,
				B: 167,
				A: 255,
			},
		},
	}
}

// GlyphCycle contains different glyphs
type GlyphCycle struct {
	Glyphs  []draw.GlyphDrawer
	Current int
}

// Next returns the next shape
func (g *GlyphCycle) Next() draw.GlyphDrawer {
	gd := g.Glyphs[g.Current%len(g.Glyphs)]
	g.Current++
	return gd
}

// NewDefaultGlyphCycle returns a new instance of GlyphCycle containing some predefined
// glyphs
func NewDefaultGlyphCycle() GlyphCycle {
	return GlyphCycle{
		Glyphs: []draw.GlyphDrawer{
			&draw.RingGlyph{},
			&draw.CrossGlyph{},
			&draw.SquareGlyph{},
			&draw.PlusGlyph{},
			&draw.TriangleGlyph{},
			&draw.PyramidGlyph{},
			&draw.CircleGlyph{},
		},
	}
}
