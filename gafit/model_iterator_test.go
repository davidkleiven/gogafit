package gafit

import "testing"

func TestSizeIterator(t *testing.T) {
	iterator := ModelIterator{
		Include: []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		MaxSize: 4,
	}

	for model := iterator.Include; model != nil; model = iterator.Next() {
		num := 0
		for i := range model {
			num += model[i]
		}

		if num > iterator.MaxSize {
			t.Errorf("Model\n%v\nexceeds max size: %d\n", model, iterator.MaxSize)
		}
	}
}

func TestModels(t *testing.T) {
	iterator := ModelIterator{
		Include: []int{0, 1, 0, 0},
		MaxSize: 8,
	}

	expectModels := [][]int{
		{0, 1, 0, 0},
		{1, 1, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 1, 0},
		{1, 0, 1, 1},
	}

	i := 0
	for model := iterator.Include; model != nil; model = iterator.Next() {
		if !AllEqualInt(model, expectModels[i]) {
			t.Errorf("Expected model #%d:\n%v\ngot\n%v\n", i, expectModels[i], model)
		}
		i++
	}
}
