package gafit

// ModelIterator iterates through all models by sequentually flipping bits
type ModelIterator struct {
	Include []int
	MaxSize int
	current int
}

// Flip flips current
func (m *ModelIterator) Flip() {
	m.Include[m.current] = (m.Include[m.current] + 1) % 2
}

// UndoLastFlip undo the prvious flip
func (m *ModelIterator) UndoLastFlip() {
	if m.current > 0 {
		m.Include[m.current-1] = (m.Include[m.current-1] + 1) % 2
	}
}

func (m *ModelIterator) size() int {
	num := 0
	for i := range m.Include {
		num += m.Include[i]
	}
	return num
}

// Next returns the next model
func (m *ModelIterator) Next() []int {
	if m.current >= len(m.Include) {
		return nil
	}
	m.Flip()

	if m.size() > m.MaxSize {
		m.Include[m.current] = 0
		m.current++
		return m.Next()
	}

	if m.size() == 0 {
		m.Flip()
	}
	m.current++
	return m.Include
}
