
export function stepSizeScale(range, values, n_steps = 5, steps = null) {
    values = values.sort((a, b) => b - a);
    if (!steps) {
      steps = [];
      for (let i = 0; i < n_steps; ++i) {
        steps.push(2 << i);
      }
    }
    return (x) => {
      for (
        let step = steps[0], i = step - 1, j = 0;
        i < values.length && j < n_steps - 1;
        j++, step += steps[j], i += step
      ) {
        if (x >= values[i]) {
          let ratio = (n_steps - 1 - j) / (n_steps - 1);
          ratio = ratio ** 1.33;
          return (
            range[0] + (range[1] - range[0]) * ratio + (x - values[i]) * 0.01
          );
        }
      }
      return range[0];
    }
}
  
export function getValuesByEntropy(words, self, layer) {
    let max_entropy, max_freq;
    max_freq = Math.max(...words.map((d) => d.frequency));
    max_entropy = Math.max(...words.map((d) => d.entropy));
    words.forEach((d) => {
      let term1, term2;
      term1 = (d.entropy / max_entropy)
      term2 = Math.log(1 + d.frequency)
    });
  
    words = words.sort((a, b) => b.value - a.value);
}

export function getWordColor(self, d) {
  if (d.prediction_score) {
    if (d.prediction_score > 0.72) {
      return self.positiveColor;
    } else if (d.prediction_score < 0.28) {
      return self.negativeColor;
    } else {
      return self.neutralColorGray;
    }
  }
  
  let tot = d.contri.pos + d.contri.neu + d.contri.neg;
  if (d.contri.neg > d.contri.pos && d.contri.neg >= d.contri.neu) {
    return self.negativeColor;
  } else if (d.contri.pos > d.contri.neg && d.contri.pos >= d.contri.neu) {
    return self.positiveColor;
  } else {
    return self.neutralColorGray;
  }
}