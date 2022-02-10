<template>
  <g class="main-DAG">
    <defs>
      <template v-for="word in words">
        <linearGradient v-for="(point, index) in word.line"
          :key="point.fill_id"
          x1="0%" y1="0%" x2="100%" y2="0%" :id="`pattern-${point.fill_id}`"
        >
          <stop offset="0%" :stop-color="contriScale(point.contri)"></stop>
          <stop offset="100%"
            :stop-color="contriScale(word.line[Math.min(n_rows, index + 1)].contri)">
          </stop>
        </linearGradient>
      </template>
    </defs>
    <g class='view' :transform="`translate(${margin.left}, ${margin.top})`">
      <g v-for="(d, index) in shades" :key="`shade${index}`" class="shade">
        <path :d="d.d" :fill="d.fill" stroke="none"></path>
      </g>
    </g>
  </g>
</template>

<script>

import * as d3 from 'd3'

const lineGenerator = (x1, x2, y1, y2, y3, y4, r1 = 0.4, r2 = 0.6) => {
  let m1 = x1+(x2-x1) * 0.5//r1
  let m2 = x1+(x2-x1) * 0.5//r2
  let x3 = x1+(x2-x1) * 0.25
  let x4 = x1+(x2-x1) * 0.75
  if (Math.abs(y4 - y2) < 30) {
    x3 = x1
    x4 = x2
    m1 = x1+(x2-x1) * 0.5
    m2 = x1+(x2-x1) * 0.5
  }
  return `M${x1} ${y1}L${x1} ${y2} L${x3} ${y2} C${m1} ${y2} ${m2} ${y4} ${x4} ${y4} L${x2} ${y4} L${x2} ${y3} L${x4} ${y3} C${m2} ${y3} ${m1} ${y1} ${x3} ${y1} z`
}
const pathGenerator = (x1, x2, y1, y2, y3, y4, r1 = 0.4, r2 = 0.6) => {
  let m1 = x1+(x2-x1) * r1
  let m2 = x1+(x2-x1) * r2
  y1 = y2
  y2 = y4
  if (Math.abs(y1 - y2) < 12) {
    let m = (y1 + y2) / 2
    y1 = y2 = m
  }
  return `M${x1} ${y1}C${m1} ${y1} ${m2} ${y2} ${x2} ${y2}`
}

export default {
  props: [ 'layout', 'width', 'height' ],
  methods: {
    check_continuity(layer, left, right) {
      if (layer == -1) return false
      if (!this.words[left].line[layer].display) {
        return false
      }
      if (!this.words[right].line[layer].display) {
        return false
      }
      if (this.words[right].line[layer].label != this.words[left].line[layer].label) {
        return false
      }
      return true
    }
  },
  computed: {
    phrases() {
      return this.layout.phrases
    },
    shades() {
      let words = this.words
      let last_contri_record = {}
      let shades = []
      for (let i = 0; i + 1 < n_rows; ++i) {
        for (let j = 1, last = 0; j <= n_cols; ++j) {
          if (j < n_cols && words[j].line[i].label == words[j - 1].line[i].label && words[j].line[i].display) {
            continue
          } /*else if (j + 1 < n_cols && words[j + 1].line[i].label == words[j - 1].line[i].label && words[j + 1].line[i].display) {
            j += 1
            continue
          } */ else {
            let curr = j - 1
            if (curr - last >= 1) {
              let w1 = widthScale(words[last].line[i].size)
              let w2 = widthScale(words[curr].line[i].size)
              let y1 = yScale(words[last].line[i].position) - w1 * 0.5 - 12
              let y2 = yScale(words[curr].line[i].position) + w2 * 0.5 + 12
              //console.log(i, last, curr)
              if (true || check_continuity(i + 1, last, curr)) {
                let w3 = widthScale(words[last].line[i + 1].size)
                let w4 = widthScale(words[curr].line[i + 1].size)
                let y3 = yScale(words[last].line[i + 1].position) - w3 * 0.5 - 12
                let y4 = yScale(words[curr].line[i + 1].position) + w4 * 0.5 + 12
                let contri = 0
                let last_contri = 0
                for (let k = last; k <= curr; ++k) {
                  contri += words[k].line[i + 1].contri
                  last_contri += words[k].line[i].contri
                }
                contri /= (curr - last + 1)
                last_contri /= (curr - last + 1)
                for (let k = last; k <= curr; ++k) {
                  last_contri_record[(i + 1) * words.length + k] = contri
                }
                for (let k = last; k <= curr; ++k) {
                  if (last_contri_record[i * words.length + k]) {
                    last_contri = last_contri_record[i * words.length + k]
                    break
                  }
                }
                let fill = contri < 0 ? contriScale.range()[0] : contriScale.range()[contriScale.range().length - 1]
                if (0 && last_contri < 0 && contri > 0) {
                  fill = 'url(#pattern-left-to-right)'
                } else if (0 && last_contri > 0 && contri < 0) {
                  fill = 'url(#pattern-right-to-left)'
                } else {
                  fill = d3.interpolateRgb('white', fill)(0.2)
                }
                shades.push({
                  y1, y2, y3, y4,
                  top: last, bottom: curr,
                  //fill: 'lightblue', //contriScale(contri),
                  fill: fill,
                  //opacity: 0.2,
                  x: i,
                  y: words[curr].line[i].label,
                })
              }
            }
            last = j
            //if (last < words.length && last > 0 && !words[last].line[i].display && words[last - 1].line[i].display && words[last].line[i].label == words[last - 1].line[i].label) {
            //  last -= 1
            //} else if (last < words.length && !words[last].line[i].display) ++last
          }
        }
      }
      let extend_shades = []
      {
        let a = 0, b = 0, c = 0
        for (let k = 0; k + 2 < n_rows; ++k) {
          while (a < shades.length && shades[a].x < k) ++a
          while (b < shades.length && shades[b].x < k + 1) ++b
          while (c < shades.length && shades[c].x < k + 2) ++c
          let cover_points = new Set()
          for (let i = a; i < b; ++i) {
            for (let j = shades[i].top; j <= shades[i].bottom; ++j) {
              cover_points.add(j)
            }
          }
          if (a < b && b < c) {
            let i = a, j = b
            while (i < b && j < c) {
              if (shades[i].bottom < shades[j].top){ ++i; continue }
              if (shades[i].top > shades[j].bottom){ ++j; continue }
              if (shades[i].top > shades[j].top || shades[i].top < shades[j].top) {
                shades[i].y3 = shades[j].y1
              }
              ++i; ++j;
            }
            i = b - 1, j = c - 1
            while (i >= a && j >= b) {
              if (shades[i].bottom < shades[j].top){ --j; continue }
              if (shades[i].top > shades[j].bottom){ --i; continue }
              if (shades[i].bottom > shades[j].bottom || shades[i].bottom < shades[j].bottom) {
                shades[i].y4 = shades[j].y2
              }
              --i; --j;
            }
          }
          for (let i = b; i < c; ++i) {
            let flag = 1
            for (let j = shades[i].top; j <= shades[i].bottom; ++j) {
              if (cover_points.has(j)) {
                flag = 0
                break
              }
            }
            if (flag) {
              let y1 = shades[i].y1
              let y2 = shades[i].y2
              let contri = words[shades[i].top].line[k + 1].contri
              let y0 = yScale((words[shades[i].top].line[k].position + words[shades[i].bottom].line[k].position) * 0.5)
              let last_contri = words[shades[i].top].line[k].contri
              let fill = contri < 0 ? contriScale.range()[0] : contriScale.range()[contriScale.range().length - 1]
              if (last_contri < 0 && contri > 0) {
                fill = 'url(#pattern-left-to-right)'
              } else if (last_contri > 0 && contri < 0) {
                fill = 'url(#pattern-right-to-left)'
              } else {
                fill = d3.interpolateRgb('white', fill)(0.2)
              }
              extend_shades.push({
                y1: y0, y2: y0, y3: y1, y4: y2,
                top: shades[i].top, bottom: shades[i].bottom,
                //fill: `lightblue`,
                fill: fill,
                //opacity: 0.2,
                x: shades[i].x - 1,
                y: shades[i].label,
              })
            }
          }
        }     
      }
      for (let i = 1; i < shades.length; ++i) {
        if (shades[i].x == shades[i - 1].x) {
          if (shades[i].y3 < shades[i - 1].y4) {
            let y = (shades[i].y3 + shades[i - 1].y4) * 0.5
            shades[i].y3 = y
            shades[i - 1].y4 = y
          }
          if (shades[i].y1 < shades[i - 1].y2) {
            let y = (shades[i].y1 + shades[i - 1].y2) * 0.5
            shades[i].y1 = y
            shades[i - 1].y2 = y
          }
        }
      }
      shades = shades.concat(extend_shades)
      shades.forEach(d => {
        d.d = lineGenerator(xScale(d.x), xScale(d.x + 1), d.y1, d.y2, d.y3, d.y4)
      })
    },
    words() {
      return this.layout.lines.map((d, k) => {
        let ret = {}
        ret.text = d.text
        let last_label = 0
        let last_display = 0
        let threshold = 1.0 / d.line.length / 2
        for (let i = 0; i < d.line.length; ++i) {
          let l = d.line[i]
          l.display = i < 3 || l.size >= threshold && (i == 0 || d.line[i - 1].display)
          if (l.display) {
            last_display = i
          }
          l.has_shade = !!l.has_shade
          l.fill_id = d.line.length * k + i
          if (i + 1 < d.line.length && l.display) {
            if (i > 0 && d.line[i - 1].show_label || i + 2 == d.line.length) {
              l.show_label = 0
            } else if (l.show_label && i - last_label > 5) {
              d.line[Math.floor((last_label + i) / 2)].show_label = 1
            }
          } else {
            l.show_label = 0
          }
          if (l.show_label) {
            last_label = i
          }
        }
        last_display += 1
        if (last_display - last_label > 5) {
          d.line[Math.floor((last_label + last_display) / 2)].show_label = 1
        }
        ret.line = d.line
        return ret
      })
    },
    n_rows() {
      return this.words[0].line.length
    },
    n_cols() {
      return this.words.length
    },
    edges() {
      return this.layout.edges
    },
    margin() {
      return { top: 25, right: 100, bottom: 25, left: 100 }
    },
    canvas_width() {
      return this.width - this.margin.left - this.margin.right 
    },
    canvas_height() {
      return this.height - this.margin.top - this.margin.bottom 
    },
    xScale() {
      return d3.scalePoint()
        .domain(this.words[0].line.map(d => d.layer))
        .rangeRound([0, this.canvas_width]) 
    },
    yScale() {
      return d3.scaleLinear()
        .domain([1, 0])
        .range([10, this.canvas_height])
    },
    widthScale() {
      return d3.scalePow().exponent(2)
        .domain(d3.extent([].concat(...this.words.map(d => d.line.map(e => e.size)))))
        .range([1.5, 9])
    },
    contriScale() {
      let values = [].concat(...this.words.map(d => d.line.map(e => e.contri)))
      values.sort((a,b) => a - b)
      let n = values.length
      let left = values[0]
      let right = values[n - 1]
      let mid = 0
      if (mid > right) mid = right - 0.005
      if (mid < left) mid = left + 0.005
      let l2 = mid - (mid - left) * 0.3
      let l1 = mid - (mid - left) * 0.7
      let r1 = mid + (right - mid) * 0.3
      let r2 = mid + (right - mid) * 0.7
      let colorScale = d3.scaleLinear()
        .domain([left, l1, l2, mid, r1, r2, right])
        .range(['rgb(26, 150, 65)', 'rgba(26, 150, 65, 0.7)', 'rgba(163,163,163, .8)', 'rgba(163,163,163, 1)', 'rgba(163,163,163, .8)', 'rgba(215,25,28,0.7)', 'rgb(215, 25, 28)'])
      return colorScale
    },
    distScale() {
      let values = [].concat(...this.words.map(d => d.line.map(e => e.contri)))
      values.sort((a,b) => a - b)
      let n = values.length
      let left = values[0]
      let right = values[n - 1]
      let mid = 0
      if (mid > right) mid = right - 0.005   
      if (mid < left) mid = left + 0.005
      let l2 = mid - (mid - left) * 0.15
      let l1 = mid - (mid - left) * 0.6
      let r1 = mid + (right - mid) * 0.15
      let r2 = mid + (right - mid) * 0.6
      let distScale = d3.scaleLinear()
        .domain([left, l1, l2, mid, r1, r2, right])
        .range([0, 1, 2, 3, 4, 5, 6, 7])
      return distScale
    },
    color_combinations() {
      let colors = { positive: `rgb(26, 150, 65)`, negative: `rgb(215, 25, 28)`, neutral: `rgb(193,193,193)` }
      let ret = []
      for (let i in colors)
        for (let j in colors) {
          ret.push({ id: `${i}-${j}`, from: colors[i], to: colors[j] })
        }
      return ret
    }
  }
}


    function paintSentenceDAG(self) {

      function LineChart() {
        const canvas = document.createElement("canvas")
        const ctx = canvas.getContext("2d")
        // Add the SVG to the page
        const svg = d3.create("svg")
            .attr("viewBox", [0, 0, width + margin.left + margin.right, height + margin.top + margin.bottom])
            .attr("width", width)
            .attr("height", height)
        const defs = svg.append('defs')
        const isalpha = val => /^[a-zA-Z]+$/.test(val)
        
        // Standard Margin Convention
        const g = svg.append("g")
            .attr('transform', `translate(${margin.left}, ${margin.top})`)
        
        // Call the x axis in a group tag
        g.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(xScale))
            .select('.domain').remove()
        
        // Append the path, bind the words, and call the line generator 
        let n_rows = words[0].line.length
        let n_cols = words.length
        let check_continuity = (layer, left, right) => {
          if (layer == -1) return false
          if (!words[left].line[layer].display) {
            return false
          }
          if (!words[right].line[layer].display) {
            return false
          }
          if (words[right].line[layer].label != words[left].line[layer].label) {
            return false
          }
          return true
        }
        
        let last_contri_record = {}
        let shades = []
        for (let i = 0; i + 1 < n_rows; ++i) {
          for (let j = 1, last = 0; j <= n_cols; ++j) {
            if (j < n_cols && words[j].line[i].label == words[j - 1].line[i].label && words[j].line[i].display) {
              continue
            } /*else if (j + 1 < n_cols && words[j + 1].line[i].label == words[j - 1].line[i].label && words[j + 1].line[i].display) {
              j += 1
              continue
            } */ else {
              let curr = j - 1
              if (curr - last >= 1) {
                let w1 = widthScale(words[last].line[i].size)
                let w2 = widthScale(words[curr].line[i].size)
                let y1 = yScale(words[last].line[i].position) - w1 * 0.5 - 12
                let y2 = yScale(words[curr].line[i].position) + w2 * 0.5 + 12
                //console.log(i, last, curr)
                if (true || check_continuity(i + 1, last, curr)) {
                  let w3 = widthScale(words[last].line[i + 1].size)
                  let w4 = widthScale(words[curr].line[i + 1].size)
                  let y3 = yScale(words[last].line[i + 1].position) - w3 * 0.5 - 12
                  let y4 = yScale(words[curr].line[i + 1].position) + w4 * 0.5 + 12
                  let contri = 0
                  let last_contri = 0
                  for (let k = last; k <= curr; ++k) {
                    contri += words[k].line[i + 1].contri
                    last_contri += words[k].line[i].contri
                  }
                  contri /= (curr - last + 1)
                  last_contri /= (curr - last + 1)
                  for (let k = last; k <= curr; ++k) {
                    last_contri_record[(i + 1) * words.length + k] = contri
                  }
                  for (let k = last; k <= curr; ++k) {
                    if (last_contri_record[i * words.length + k]) {
                      last_contri = last_contri_record[i * words.length + k]
                      break
                    }
                  }
                  let fill = contri < 0 ? contriScale.range()[0] : contriScale.range()[contriScale.range().length - 1]
                  if (0 && last_contri < 0 && contri > 0) {
                    fill = 'url(#pattern-left-to-right)'
                  } else if (0 && last_contri > 0 && contri < 0) {
                    fill = 'url(#pattern-right-to-left)'
                  } else {
                    fill = d3.interpolateRgb('white', fill)(0.2)
                  }
                  shades.push({
                    y1, y2, y3, y4,
                    top: last, bottom: curr,
                    //fill: 'lightblue', //contriScale(contri),
                    fill: fill,
                    //opacity: 0.2,
                    x: i,
                    y: words[curr].line[i].label,
                  })
                }
              }
              last = j
              //if (last < words.length && last > 0 && !words[last].line[i].display && words[last - 1].line[i].display && words[last].line[i].label == words[last - 1].line[i].label) {
              //  last -= 1
              //} else if (last < words.length && !words[last].line[i].display) ++last
            }
          }
        }
        // console.log(shades)
        let extend_shades = []
        {
          let a = 0, b = 0, c = 0
          for (let k = 0; k + 2 < n_rows; ++k) {
            while (a < shades.length && shades[a].x < k) ++a
            while (b < shades.length && shades[b].x < k + 1) ++b
            while (c < shades.length && shades[c].x < k + 2) ++c
            let cover_points = new Set()
            for (let i = a; i < b; ++i) {
              for (let j = shades[i].top; j <= shades[i].bottom; ++j) {
                cover_points.add(j)
              }
            }
            if (a < b && b < c) {
              let i = a, j = b
              while (i < b && j < c) {
                if (shades[i].bottom < shades[j].top){ ++i; continue }
                if (shades[i].top > shades[j].bottom){ ++j; continue }
                if (shades[i].top > shades[j].top || shades[i].top < shades[j].top) {
                  shades[i].y3 = shades[j].y1
                }
                ++i; ++j;
              }
              i = b - 1, j = c - 1
              while (i >= a && j >= b) {
                if (shades[i].bottom < shades[j].top){ --j; continue }
                if (shades[i].top > shades[j].bottom){ --i; continue }
                if (shades[i].bottom > shades[j].bottom || shades[i].bottom < shades[j].bottom) {
                  shades[i].y4 = shades[j].y2
                }
                --i; --j;
              }
            }
            for (let i = b; i < c; ++i) {
              let flag = 1
              for (let j = shades[i].top; j <= shades[i].bottom; ++j) {
                if (cover_points.has(j)) {
                  flag = 0
                  break
                }
              }
              if (flag) {
                let y1 = shades[i].y1
                let y2 = shades[i].y2
                let contri = words[shades[i].top].line[k + 1].contri
                let y0 = yScale((words[shades[i].top].line[k].position + words[shades[i].bottom].line[k].position) * 0.5)
                let last_contri = words[shades[i].top].line[k].contri
                let fill = contri < 0 ? contriScale.range()[0] : contriScale.range()[contriScale.range().length - 1]
                if (last_contri < 0 && contri > 0) {
                  fill = 'url(#pattern-left-to-right)'
                } else if (last_contri > 0 && contri < 0) {
                  fill = 'url(#pattern-right-to-left)'
                } else {
                  fill = d3.interpolateRgb('white', fill)(0.2)
                }
                extend_shades.push({
                  y1: y0, y2: y0, y3: y1, y4: y2,
                  top: shades[i].top, bottom: shades[i].bottom,
                  //fill: `lightblue`,
                  fill: fill,
                  //opacity: 0.2,
                  x: shades[i].x - 1,
                  y: shades[i].label,
                })
              }
            }
          }     
        }
        for (let i = 1; i < shades.length; ++i) {
          if (shades[i].x == shades[i - 1].x) {
            if (shades[i].y3 < shades[i - 1].y4) {
              let y = (shades[i].y3 + shades[i - 1].y4) * 0.5
              shades[i].y3 = y
              shades[i - 1].y4 = y
            }
            if (shades[i].y1 < shades[i - 1].y2) {
              let y = (shades[i].y1 + shades[i - 1].y2) * 0.5
              shades[i].y1 = y
              shades[i - 1].y2 = y
            }
          }
        }
        shades = shades.concat(extend_shades)
        shades.forEach(d => {
          d.d = lineGenerator(xScale(d.x), xScale(d.x + 1), d.y1, d.y2, d.y3, d.y4)
        })
        
        let shade = g.selectAll('.shade')
          .tatta(shades).enter()
          .append('g')
          .attr('class', 'shade')
          //.style('opacity', d => d.opacity)
      
        shade
          .append('path')
          .attr('d', d => d.d)
          .style('stroke', 'none')
          .style('fill', d => d.fill)
        
      
        const line = g.selectAll(".line")
          .tatta(words)
          .enter().append("g")
          .attr("class", "line")
        
        g.append("text")
          .attr("x", -50)
          .attr("y", -8)
          .style("font-size", "13px")
          .style("font-family", "Roboto, san-serif")
          .text('word in sentence')
        
        line.append("text")
          .attr("x", 0)
          .attr("y", d => yScale(d.line[0].position) + 5)
          .attr("text-anchor", "end")
          .style("font-size", "13px")
          .style("font-family", "Roboto, san-serif")
          .text(d => d.text)
        
        let top_size = words.map(d => d.line[d.line.length - 1].size).sort((a, b) => b - a)
        top_size = top_size.length > 3 ? top_size[3] : top_size[top_size.length - 1] - 1e-6
                                                      
        const phrase_data = phrases.filter(d => d.type == 'phrase' && d.layer < words[0].line.length)
                          .filter(d => {
                              let x = (xScale(d.layer - 1) * 3 + xScale(d.layer)) / 4
                              let k = 0
                              for (let i = d.top; i < d.bottom; ++i) {
                                if (words[i].line[d.layer - 1].label != words[i + 1].line[d.layer - 1].label) {
                                  k = i
                                  break 
                                }
                              }
                              return (words[k].line[d.layer].display && words[k + 1].line[d.layer].display)
                          })
                          
        
        const relation_data = phrases
          .filter(d => d.type == 'relation')
          .filter(d => words[d.top].line[d.layer].display && words[d.bottom].line[d.layer].display)
        
        const phrase = g.selectAll(".phrase")
          .tatta(phrase_data)
          .enter()
          .append("g")
          .attr("class", "phrase")
        
        phrase.append('path')
          .attr('d', d => {
            let x = (xScale(d.layer - 1) * 3 + xScale(d.layer)) / 4
            let k = 0
            for (let i = d.top; i < d.bottom; ++i) {
              if (words[i].line[d.layer - 1].label != words[i + 1].line[d.layer - 1].label) {
                k = i
                break
              }
            }
            if (!words[d.top].line[d.layer - 1].display || !words[k+1].line[d.layer - 1].display) return ''
            words[k].line[d.layer - 1].has_rseg = 1
            words[k + 1].line[d.layer - 1].has_rseg = 1
            let y = yScale((words[k].line[d.layer - 1].position + words[k + 1].line[d.layer - 1].position) / 2)
                          //words[k].line[d.layer].position + words[k + 1].line[d.layer].position) / 4)
            let y1 = yScale(words[k].line[d.layer - 1].position)
            let y2 = yScale(words[k + 1].line[d.layer - 1].position)
            return `M${x-20} ${y1} Q${x-10} ${y-2} ${x} ${y-2} M${x-20} ${y2} Q${x-10} ${y+2} ${x} ${y+2}`
          })
          .style('fill', 'none')
          .style('stroke', 'rgb(27, 30, 35)')
          .style('opacity', .6)
          .style('stroke-width', '2px')
          
        phrase.append('path')
          .attr('transform', d => {
            let x = (xScale(d.layer - 1) * 3 + xScale(d.layer)) / 4
            let k = 0
            for (let i = d.top; i < d.bottom; ++i) {
              if (words[i].line[d.layer - 1].label != words[i + 1].line[d.layer - 1].label) {
                k = i
                break
              }
            }
            if (!words[d.top].line[d.layer - 1].display || !words[k + 1].line[d.layer - 1].display) return ''
            let y = yScale((words[k].line[d.layer - 1].position + words[k + 1].line[d.layer - 1].position) / 2)
            return `translate(${x+3},${y})`
          })
          .attr('d', d => `M${1 * 6} ${0} L${-0.5 * 6} ${0.866 * 6} L${-0.5 * 6} ${-0.866 * 6} Z`)
          .style('fill', 'rgb(27, 30, 35)')
          .style('stroke', 'none')
          .style('opacity', .6)
          .style('stroke-width', '2px')
        
        let rsegs = []
        for (let i = 1; i + 1 < n_rows; ++i) {
          for (let j = 0; j < n_cols; ++j) {
            //let a = distScale(words[j].line[i].contri)
            //let b = distScale(words[j].line[Math.max(0, i - 3)].contri)
            let a = words[j].line[i].contri
            let b = words[j].line[0].contri
            let c = words[j].line[n_rows - 2].contri
            if (!words[j].line[i].display) continue
            let left = Math.abs(distScale(b) - distScale(a))
            let right = Math.abs(distScale(c) - distScale(b))
            // console.log(left, right, a, b, c)
            rsegs.push({
              x: i, y: j,
              d: distScale(words[j].line[i - 1].contri) - distScale(words[j].line[i + 1].contri),
              left: words[j].line[i - 1].contri,
              right: words[j].line[i + 1].contri,
              direction: right < left ? 1 : 0
            })
          }
        }
        rsegs = rsegs.sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).filter(d => Math.abs(d.d) > 0.015)
        for (let i = 0; i < rsegs.length; ++i) {
          rsegs[i].deleted = false
          if (rsegs[i].x >= words[0].line.length - 3) {
            rsegs[i].deleted = true
            continue
          }
          for (let j = 0; j < i; ++j) {
            if ((rsegs[i].y == rsegs[j].y || Math.abs(rsegs[i].x - rsegs[j].x) + Math.abs(rsegs[i].y - rsegs[j].y) <= 2) && !rsegs[j].deleted) {
              rsegs[i].deleted = true
            }
          }
        }
        rsegs = rsegs.filter(d => !d.deleted).slice(0, 7)
        rsegs = rsegs.map(d => ({ x: xScale(d.x), y: yScale(words[d.y].line[d.x].position), row: d.y, col: d.x, delta: d.d, direction: d.direction }))
        // console.log('rsegs', rsegs)
          /*
        rsegs.forEach(d => {
          let label = words[d.row].line[d.col].label
          let weight = edges[d.col - 1][label].slice(1, 1 + words.length)
          let pl = Math.max(1, d.col - 3)
          let plabel = words[d.row].line[pl].label
          // console.log(pl, plabel, edges[pl].length)
          let pweight = edges[pl - 1][plabel].slice(1, 1 + words.length)
          d.changes = weight.map((e, i) => [i, e - pweight[i]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 4)
          d.changes = d.changes.sort((a, b) => a[0] - b[0]).map(e => [words[e[0]].text, e[1]])
          d.add_changes = d.changes.filter(e => e[1] > 0)
          d.tot_add = 0
          for (let e of d.add_changes) d.tot_add += Math.abs(e[1])
          d.drop_changes = d.changes.filter(e => e[1] < 0)
          d.tot_drop = 0
          for (let e of d.drop_changes) d.tot_drop += Math.abs(e[1])
        })*/
        for (let d of rsegs){ words[d.row].line[d.col].has_rseg = 1 }
        for (let i = 1; i < words.length; ++i) {
          let last = -1
          for (let j = 0; j < words[0].line.length; ++j) {
            if (j - last > 4) {
              words[i].line[j].show_label = 1
            }
            if (words[i].line[j].show_label) {
              if (j + 2 >= words[i].line.length) {
                words[i].line[j].show_label = 0
              } else if (j < last) {
                words[i].line[j].show_label = 0
              } else if (j - last <= 3) {
                words[i].line[j].show_label = 0
                if (last + 3 < words[i].line.length) {
                  words[i].line[last + 3].show_label = 1
                }
                last = last + 3
              } else if (words[i].line[j].has_rseg) {
                words[i].line[j].show_label = 0
                if (j + 1 < words[i].line.length) {
                  words[i].line[j + 1].show_label = 1
                }
              } else if (words[i].line[j + 1].has_rseg) {
                words[i].line[j].show_label = 0
                if (j + 2 < words[i].line.length) {
                  words[i].line[j + 2].show_label = 1
                }
              } else {
                last = j
              }
              //if (words[i].line[j + 1].has_rseg);
              //if (words[i - 1].line[j].show_label);
            }
          }
        }
        let annotations = []
        
        for (let d of relation_data) {
          let k = -1, max_delta = 0
          for (let i = d.top; i < d.bottom; ++i) {
            let delta =
                Math.abs(words[i + 1].line[d.layer].position - words[i].line[d.layer].position) + 
                Math.abs(words[i + 1].line[d.layer + 1].position - words[i].line[d.layer + 1].position)
            if (delta > max_delta) {
              max_delta = delta
              k = i
            }
          }
          let top = Math.max(words[k + 1].line[d.layer].position, words[k + 1].line[d.layer + 1].position)
          let bottom = Math.min(words[k].line[d.layer].position, words[k].line[d.layer + 1].position)
          annotations.push({
            x: xScale(d.layer) + 25,
            y: yScale((top + bottom) / 2) + 5,
            fill: 'gray',
            'text-anchor': 'start',
            'font-size': 15,
            has_line: false,
            text: d.text,
          })
        }
        
        const annotation = g.selectAll(".annotation")
          .tatta(annotations)
          .enter()
          .append("g")
          .attr("class", "annotation")
        
        let represent = g.selectAll('.represent')
          .tatta(rsegs).enter()
          .append('g')
          .attr('class', 'represent')
          .attr('transform', d => `translate(${d.x},${d.y})`)
        
        represent.append('rect')
          .attr('x', -9)
          .attr('y', d => -5)
          .attr('width', 18)
          .attr('height', d => 10)
          .attr('rx', 2)
          .attr('ry', 2)
          .style('stroke', d => {
            let p = distScale(words[d.row].line[d.col].contri)
            let n = contriScale.range().length
            if (p < n / 2) {
              return contriScale.range()[Math.floor(p)]
            } else {
              return contriScale.range()[Math.min(n - 1, Math.ceil(p))]
            }
          })
          .style('stroke-width', 1)
          .style('fill', d => contriScale(words[d.row].line[d.col + 1].contri))
          
        represent.append('rect')
          .attr('x', -9)
          .attr('y', d => -5)
          .attr('width', 9)
          .attr('height', d => 10)
          .attr('rx', 2)
          .attr('ry', 2)
          .style('stroke', 'none')
          .style('fill', d => contriScale(words[d.row].line[d.col - 1].contri))
        /*
        represent.append('path')
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .attr('d', d => {
            let r = 6.5, ret
            if (d.direction == 1) {
              ret = `M${1 * r} ${0} L${-0.5 * r} ${0.866 * r} L${-0.5 * r} ${-0.866 * r} Z`
            } else {
              ret = `M${-1 * r} ${0} L${0.5 * r} ${0.866 * r} L${0.5 * r} ${-0.866 * r} Z`
            }
            return ret
          })
          .style('stroke', 'rgb(27, 30, 35)')
          .style('stroke-width', 1)
          .style('fill', d => d.delta < 0 ? contriScale.range()[1] : contriScale.range()[contriScale.range().length - 2])*/
        represent
          .on('mouseover', function(d){
          // console.log(d)
          let color = d.delta < 0 ? contriScale.range()[0] : contriScale.range()[contriScale.range().length - 1]
          if (d.add_changes.length > 0) {
            d3.select(this)
              .append('rect')
              .attr('class', 'attached')
              //.attr('transform', d => `translate(${d.x},${d.y})`)
              .attr('x', -150)
              .attr('y', -50)
              .attr('width', 305)
              .attr('height', 42)
              .style('stroke', 'none')
              .style('fill', 'rgba(255,255,255,0.7)')
            d3.select(this)
              .append('path')
              .attr('class', 'attached')
              //.attr('transform', d => `translate(${d.x},${d.y})`)
              .attr('d', d => {
                let x1 = -80
                let x2 = 0
                let y1 = -30
                let y2 = 0
                let ret = `M${x1} ${y1} C${(x1+x2)*0.5} ${y1} ${(x1+x2)*0.5} ${y2} ${x2} ${y2}`
                // console.log(ret)
                return ret
              })
              .style('stroke-dasharray', '5, 5')
              .style('stroke', color)
              .style('stroke-width', '3px')
              .style('fill', 'none')
            
            d3.select(this)
              .append('text')
              .attr('class', 'attached')
              //.attr('transform', d => `translate(${d.x},${d.y})`)
              .attr('x', -80).attr('y', -35)
              .style('fill', color)
              .style("font-size", "13px")
              .style('font-family', 'Roboto, san-serif')
              .text(`+ ${d.add_changes.map(e => e[0]).filter(e => isalpha(e)).join(',')}`) 
          }
          if (d.drop_changes.length > 0) {
            d3.select(this)
              .append('rect')
              .attr('class', 'attached')
              //.attr('transform', d => `translate(${d.x},${d.y})`)
              .attr('x', -150)
              .attr('y', 8)
              .attr('width', 305)
              .attr('height', 42)
              .style('stroke', 'none')
              .style('fill', 'rgba(255,255,255,0.7)')
            d3.select(this)
              .append('path')
              .attr('class', 'attached')
              //.attr('transform', d => `translate(${d.x},${d.y})`)
              .attr('d', d => {
                let x1 = 80
                let x2 = 0
                let y1 = 30
                let y2 = 0
                let ret = `M${x1} ${y1} C${(x1+x2)*0.5} ${y1} ${(x1+x2)*0.5} ${y2} ${x2} ${y2}`
                // console.log(ret)
                return ret
              })
              .style('stroke-dasharray', '5, 5')
              .style('stroke', 'gray')
              .style('stroke-width', '3px')
              .style('fill', 'none')
            
            d3.select(this)
              .append('text')
              .attr('class', 'attached')
              //.attr('transform', d => `translate(${d.x},${d.y})`)
              .attr('x', 60).attr('y', 45)
              .style('fill', 'gray')
              .style("font-size", "13px")
              .style('font-family', 'Roboto, san-serif')
              .text(`- ${d.drop_changes.map(e => e[0]).filter(e => isalpha(e)).join(',')}`) 
          }
          d3.select(this)
            .append('text')
            .attr('class', 'attached')
            .attr('transform', d => `translate(${0},${- 10})`)
            .style('fill', color)
            .style("font-size", "13px")
              .style('font-family', 'Roboto, san-serif')
            .text(`Prediction: ${d.delta >= 0 ? '' : '+'}${Number(-d.delta * 100).toFixed(2)}%`)
        })
          .on('mouseout', function(d) {
          d3.select(this).selectAll('.attached').remove()
        })
        
        annotation
          .append("text")
          .attr("x", d => d.x)
          .attr("y", d => d.y)
          .attr("fill", d => d.fill)
          .attr('text-anchor', d => d['text-anchor'])
          .attr('font-size', d => d['font-size'])
          .style('font-family', 'Roboto, san-serif')
          .text(d => d.text)
          
        const relation = g.selectAll(".relation")
          .tatta(relation_data)
          .enter()
          .append("g")
          .attr("class", "relation")
        
        relation
          .append("path")
          .attr("d", d => {
            let x1 = xScale(d.layer)
            let y1 = yScale(d.y1)
            let y2 = yScale(d.y2)
            if (y1 > y2){ let t = y1; y1 = y2; y2 = t; }
            // console.log('relation', d.top, d.bottom, words[d.top].line[d.layer].label, words[d.bottom].line[d.layer].label)
            return `M${x1},${y1} Q${x1 + (y2 - y1) / 2},${(y1+y2)/2},${x1},${y2}`
          })
          .style("stroke", "gray")
          .style("stroke-width", 4)
          .style("stroke-dasharray", "8,8")
          .style("fill", "none")
          .style("opacity", 0.3)
          
        let gradient = defs.append('linearGradient')
          .attr('id', `pattern-left-to-right`)
          .attr('x1', '0%')
          .attr('y1', '0%')
          .attr('x2', '100%')
          .attr('y2', '0%')

        gradient.append('stop')
          .attr('offset', '0%')
          .attr('stop-color', d3.interpolateRgb('white', contriScale.range()[contriScale.range().length - 1])(0.2))

        gradient.append('stop')
          .attr('offset', '100%')
          .attr('stop-color', d3.interpolateRgb('white', contriScale.range()[0])(0.2))
          
        gradient = defs.append('linearGradient')
            .attr('id', `pattern-right-to-left`)
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '100%')
            .attr('y2', '0%')
      
        gradient.append('stop')
          .attr('offset', '0%')
          .attr('stop-color', d3.interpolateRgb('white', contriScale.range()[contriScale.range().length - 1])(0.2))

        gradient.append('stop')
          .attr('offset', '100%')
          .attr('stop-color', d3.interpolateRgb('white', contriScale.range()[0])(0.2))
          
        line.each(function(d, index) {
          let segments = []
          let labels = []
          let end_point = d.line.length - 1
          for (let i = 0; i + 1 < d.line.length; ++i) {
            d.line[i].has_shade = false
            let w1 = widthScale(d.line[i].size)
            let w2 = widthScale(d.line[i + 1].size)
            if (!d.line[i].display) {
              end_point = i
              break
            } else if (!d.line[i + 1].display) {
              w2 = 0.5
            }
            let x1, x2, y1, y2, y3, y4
          
            x1 = xScale(d.line[i].layer)
            x2 = xScale(d.line[i + 1].layer)
            y1 = yScale(d.line[i].position) - w1 * 0.5
            y3 = yScale(d.line[i + 1].position) - w2 * 0.5
            
            let linearGradient = defs.append('linearGradient')
              .attr('id', `pattern-${d.line[i].fill_id}`)
              .attr('x1', '0%')
              .attr('y1', '0%')
              .attr('x2', '100%')
              .attr('y2', '0%')
      
            linearGradient.append('stop')
              .attr('offset', '0%')
              .attr('stop-color', contriScale(d.line[i].contri))
      
            linearGradient.append('stop')
              .attr('offset', '100%')
              .attr('stop-color', contriScale(d.line[i + 1].contri))
            
            if (d.line[i].show_label && isalpha(d.text)) {
              ctx.font = "13px Roboto, san-serif"
              let tw = ctx.measureText(d.text).width
              let t = Math.min(y1, y3)
              let b = Math.max(y1 + w1, y3 + w2)
              let delta = Math.abs(t - b) < 8 ? 3 : 1
              let x3 = Math.max(x1, (x1 + x2) / 2 - tw * 0.5 - delta)
              let x4 = Math.min(x2, (x1 + x2) / 2 + tw * 0.5 + delta)
              defs.append('clipPath')
                .attr('id', `clipping-${d.line[i].fill_id}`)
              .append('path')
                .attr('d', `M${x1} ${t} L${x1} ${b} L${x3} ${b} L${x3} ${t} M${x4} ${t} L${x4} ${b} L${x2} ${b} L${x2} ${t}`)
              
              let path = pathGenerator(x1, x2, y1, y1 + w1, y3, y3 + w2)
              svg.append('defs')
                .append('path')
                .attr('id', `labelbaseline-${d.line[i].fill_id}`)
                .attr('d', path)
            }
              
            segments.push({
              d: lineGenerator(x1, x2, y1, y1 + w1, y3, y3 + w2),//, r1, r2),
              opacity: 0.8,
              fill_id: d.line[i].fill_id,
              show_label: d.line[i].show_label,
              x: -1,
              y: -1,
            })
          }
          
          let seg = d3.select(this).selectAll('.segment')
            .tatta(segments).enter()
            .append('g')
            .attr('class', 'segment')
          
          seg
            .append('path')
            .attr('d', d => d.d)
            .style('clip-path', d => d.show_label ? `url(#clipping-${d.fill_id})` : 'none')
            .style('fill', d => `url(#pattern-${d.fill_id})`)
            .style('stroke', 'none')
            .style('opacity', d => d.opacity)
            
          seg.filter(d => d.show_label)
            .append('text')
            .append('textPath')
            .attr('href', d => `#labelbaseline-${d.fill_id}`)
            .attr('fill', 'rgb(27, 30, 35)')
            .style('font-size', '13px')
            .style('font-family', 'Roboto, san-serif')
            .attr('text-anchor','middle')
            .text(d.text)
            .attr('startOffset', '50%')
            .text(d.text)
            
          let contri = d.line[end_point].contri
          if (widthScale(d.line[end_point].size) < 4 && end_point + 1 < d.line.length) {
            d3.select(this)
              .append('circle')
              .attr('cx', xScale(d.line[end_point].layer))
              .attr('cy', yScale(d.line[end_point].position))
              .attr('r', 3)
              //.attr('r', Math.max(2, widthScale(d.line[end_point].size) / 2))
              .style('fill', contriScale(contri))//'#ccc')
              .style('stroke', contriScale(contri))
              .style('stroke-width', 1.5)
          }
          
          d.end_y = yScale(d.line[end_point].position) + 3
          if (index > 0 && words[index - 1].end_y - d.end_y < 10) {
            d.end_y = words[index - 1].end_y - 10
          }

          if (end_point != d.line.length - 1) {
            d3.select(this)
              .append('text')
              .attr('x', xScale(d.line[end_point].layer) + 8)// + (end_point + 1 < d.line.length ? 5 : 0))
              .attr('y', yScale(d.line[end_point].position) + 3)
              .attr('fill', end_point + 1 < d.line.length ? contriScale(contri) : 'rgb(27, 30, 35)')
              //.attr('fill', 'rgb(27, 30, 35)')
              .style('font-size', '13px')
              .style('font-family', 'Roboto, san-serif')
              .attr('text-anchor','start')
              .text(d.text)
          }
          
          d3.select(this)
            .append('text')
            .attr('x', xScale(d.line[d.line.length - 1].layer) + 8)// + (end_point + 1 < d.line.length ? 5 : 0))
            .attr('y', yScale(d.line[end_point].position) + 3)
            .attr('fill', end_point != d.line.length - 1 ? 'rgba(27, 30, 35, 0.5)' : 'rgba(27, 30, 35, 1)')
            //.attr('fill', 'rgb(27, 30, 35)')
            .style('font-size', '13px')
            .style('font-family', 'Roboto, san-serif')
            .attr('text-anchor','start')
            .text(d.text)
            
        })
        
        return svg.node().innerHTML
      }

      let dom = document.getElementById('network-svg')
      let chart = LineChart()
      dom.style.opacity = 0

      setTimeout(() => {
        dom.innerHTML = chart
        dom.style.opacity = 1
      }, 2000)
      
    }
</script>

<style scoped>

</style>
