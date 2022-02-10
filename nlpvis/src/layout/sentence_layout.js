import * as d3 from 'd3'

function forceCollide() {
  let nodes
  const padding = 4

  function force(alpha) {
    const quad = d3.quadtree(nodes, d => d.x, d => d.y);
    for (const d of nodes) {
      quad.visit((q, x1, y1, x2, y2) => {
        let updated = false;
        if (q.data && q.data !== d) {
          let x = d.x - q.data.x,
            y = d.y - q.data.y,
            xSpacing = padding + (q.data.width + d.width) / 2,
            ySpacing = padding + (q.data.height + d.height) / 2,
            absX = Math.abs(x),
            absY = Math.abs(y),
            l,
            lx,
            ly;

          if (absX < xSpacing && absY < ySpacing) {
            l = Math.sqrt(x * x + y * y);

            lx = (absX - xSpacing) / l;
            ly = (absY - ySpacing) / l;

            // the one that's barely within the bounds probably triggered the collision
            if (Math.abs(lx) > Math.abs(ly)) {
              lx = 0;
            } else {
              ly = 0;
            }
            d.x -= x *= lx;
            d.y -= y *= ly;
            q.data.x += x;
            q.data.y += y;

            updated = true;
          }
        }
        return updated;
      });
    }
  }

  force.initialize = _ => nodes = _;

  return force;
}

let testmode = 0

async function paintSampleComposition(self) {
  let positiveColor = self.positiveColor //'rgb(26,150,65)'
  let neutralColor = self.neutralColor //'rgb(163,163,163)'
  let negativeColor = self.negativeColor //d3.interpolateRgb('white', 'rgb(215,25,28)')(0.8)
  if (testmode) {
    //positiveColor = 'rgb(172, 196, 219)'
    //negativeColor = 'rgb(172, 196, 221)'
  }
  const raw = self.layout
  const rawPhraseData = raw.phrases
  const data = raw.lines.map((d, k) => {
    let ret = {}
    ret.text = d.text
    let last_label = 0
    let last_display = 0
    for (let i = 0; i < d.line.length; ++i) {
        let l = d.line[i]
        // l.display = 1 || i < 3 || l.size >= threshold && (i == 0 || d.line[i - 1].display)
        //l.size = Math.abs(l.contri)
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
    ret.end_point = last_display
    if (last_display + 1 < d.line.length) {
      ret.end_point += 1
    }
    last_display += 1
    if (last_display - last_label > 5) {
        d.line[Math.floor((last_label + last_display) / 2)].show_label = 1
    }
    ret.len = d.len
    ret.line = d.line
    return ret
  })

  for (let i = 0; i < data.length; ++i) {
    data[i].pos = 0
    if (data[i].text == '') {
      data[i].text = data[i - 1].text
      data[i].pos = data[i - 1].pos + data[i - 1].len
    }
    console.log(data[i].text, data[i].line.map(d => d.show_label))
    console.log(data[i].text, data[i].line.map(d => d.display))
  }

  const n_rows = data[0].line.length
  const n_cols = data.length
  const checkContinuity = (layer, left, right) => {
    if (layer == -1) return false
    if (!data[left].line[layer].display) {
      return false
    }
    if (!data[right].line[layer].display) {
      return false
    }
    if (data[right].line[layer].label != data[left].line[layer].label) {
      return false
    }
    return true
  }

  const getShade = (data) => {
    let shades = []
    for (let i = 0; i + 1 < n_rows; ++i) {
      for (let j = 1, last = 0; j <= n_cols; ++j) {
        if (j < n_cols && data[j].line[i].label == data[j - 1].line[i].label) {
          continue
        } else {
          let curr = j - 1
          const trim = (top, bottom, index) => {
            while (top > bottom && (index == 0 || !data[top].line[index - 1].display)) --top
            while (top > bottom && (index == 0 || !data[bottom].line[index - 1].display)) ++bottom
            let display = 0
            let tk = -1
            for (let k = bottom; k <= top; ++k) {
              if (data[k].line[index].display) {
                display = 1
                tk = k
                break
              }
            }
            return [top, bottom, tk]
          }
          let top, bottom, tk
          [top, bottom, tk] = trim(curr, last, i)
          if (top - bottom >= 1 && tk != -1) {
            let w1 = widthScale(data[bottom].line[i].contri)
            let w2 = widthScale(data[top].line[i].contri)
            let y1 = yScale(data[bottom].line[i].position) - w1 * 0.5 - 8
            let y2 = yScale(data[top].line[i].position) + w2 * 0.5 + 8
            //console.log(i, bottom, top)
            if (true || checkContinuity(i + 1, bottom, top)) {
              let rt, rb, rk
              [rt, rb, rk] = trim(curr, last, i + 1)
              let w3 = widthScale(data[rb].line[i + 1].contri)
              let w4 = widthScale(data[rt].line[i + 1].contri)
              let y3 = yScale(data[rb].line[i + 1].position)
              let y4 = yScale(data[rt].line[i + 1].position)

              if (rb != rt) {
                y3 = y3 - w3 * 0.5 - 8
                y4 = y4 + w4 * 0.5 + 8
              }
              let contri = data[tk].line[i + 1].contri
              let last_contri = data[tk].line[i].contri
              for (let k = bottom; k <= top; ++k) {
                if (!data[k].line[i].display) {
                  continue
                }
                let t = data[k].line[i + 1].contri
                if (Math.abs(t) > Math.abs(contri)) {
                  contri = t
                }
                t = data[k].line[i].contri
                if (Math.abs(t) > Math.abs(last_contri)) {
                  last_contri = t
                }
              }

              let c2 = contriScale(contri)
              let c1 = contriScale(last_contri)
              // console.log(i, top, bottom, c1, c2)
              c1 = getColorName(c1)
              c2 = getColorName(c2)
              // console.log(c1, c2)
              let fill = `url(#pattern-${c1}-to-${c2})`
              // console.log(i, top, bottom, contriThres, last_contri, contri)
              shades.push({
                y1, y2, y3, y4,
                top: bottom, bottom: top,
                c1: c1, c2: c2,
                fill: fill,
                x: i,
                y: data[top].line[i].label,
              })
            }
          }
          last = j
        }
      }
    }
    return shades
  }

  const getExtendShade = (shades, data) => {
    let extend_shades = []
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
          if (shades[i].bottom < shades[j].top) { ++i; continue }
          if (shades[i].top > shades[j].bottom) { ++j; continue }

          if (shades[i].c1 == shades[j].c2 && shades[i].c2 != shades[i].c1) {
            shades[i].c2 = shades[j].c1 = shades[j].c2
            shades[i].fill = `url(#pattern-${shades[i].c1}-to-${shades[i].c2})`
            shades[j].fill = `url(#pattern-${shades[j].c1}-to-${shades[j].c2})`
          } else if (shades[i].c2 != shades[j].c1) {
            shades[i].c2 = shades[j].c1
            shades[i].fill = `url(#pattern-${shades[i].c1}-to-${shades[i].c2})`
          }
          if (shades[i].top > shades[j].top || shades[i].top < shades[j].top) {
            shades[i].y3 = shades[j].y1
          }
          ++i; ++j;
        }
        i = b - 1, j = c - 1
        while (i >= a && j >= b) {
          if (shades[i].bottom < shades[j].top) { --j; continue }
          if (shades[i].top > shades[j].bottom) { --i; continue }
                    /*
                    if (shades[i].c1 == shades[j].c2 && shades[i].c2 != shades[i].c1) {
                        shades[i].c2 = shades[j].c1 = shades[j].c2
                        shades[i].fill = `url(#pattern-${shades[i].c1}-to-${shades[i].c2})`
                        shades[j].fill = `url(#pattern-${shades[j].c1}-to-${shades[j].c2})`
                    } else*/ if (shades[i].c2 != shades[j].c1) {
            shades[i].c2 = shades[j].c1
            shades[i].fill = `url(#pattern-${shades[i].c1}-to-${shades[i].c2})`
          }
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
          let c2 = shades[i].c1
          let y0 = yScale((data[shades[i].bottom].line[k].position + data[shades[i].bottom].line[k].position) * 0.5)
          let last_contri = data[shades[i].top].line[k].contri
          let c1 = getColorName(contriScale(last_contri))
          let fill = `url(#pattern-${c1}-to-${c2})`
          extend_shades.push({
            y1: y0, y2: y0, y3: y1, y4: y2,
            top: shades[i].top, bottom: shades[i].bottom,
            //fill: `lightblue`,
            fill: fill,
            x: shades[i].x - 1,
            y: shades[i].label,
          })
        }
      }
    }
    return extend_shades
  }
  const canvas = document.createElement("canvas")
  const ctx = canvas.getContext("2d")
  const label_size = testmode ? "32px" : "20px"
  const label_font = "Roboto, san-serif"
  ctx.font = label_size + ' ' + label_font
  const measure = (text) => {
    const ret = ctx.measureText(text)
    const width = ret.actualBoundingBoxRight + ret.actualBoundingBoxLeft
    const height = ret.actualBoundingBoxAscent + 1 //ret.actualBoundingBoxDescent
    return [width, height]
  }

  const max_text_len = 16
  const text_len = Math.max(...data.map(d => measure(d.text.slice(0, max_text_len))[0]))
  const margin = ({ top: 20, right: 10 + text_len, bottom: 20, left: 20 + text_len })
  const width = self.DAG.width - margin.left - margin.right
  const height = self.DAG.height - margin.top - margin.bottom

  data.forEach((d, index) => {
    for (let i = 1; i < d.line.length; ++i) {
      if (testmode) {
        d.line[i].display = 1
        if (index <= 6) {
          d.line[i].position += 0.06
          d.line[i].position = Math.min(d.line[i].position, d.line[i - 1].position)
        }
        if (index == 1 && i == 3) d.line[i].position -= 0.02
        if (index == 1 && i == 4) d.line[i].position += 0.06
        if (index == 3 && i == 2) d.line[i].position -= 0.02
        if (index == 3 && i >= 3) d.line[i].position = d.line[i - 1].position
        if (index == 9 && i >= 3) d.line[i].position = d.line[10].position
        //if (index == 1 && i == 3) d.line[i].position = data[index + 1].line[i].position + 0.03
        if (index > 6 && i >= 6) d.line[i].label += 1
        if ((d.text == 'the' || d.text == 'a') && i > 8) {
          d.line[i].display = 0
          d.end_point = 9
        }
        if ((d.text == "was") && i > 9) {
          d.line[i].display = 0
          d.end_point = 10
        }
        if ((d.text == "'") && i > 7) {
          d.line[i].display = 0
          d.end_point = 8
        }
        if (d.text == 'the' && i == 4) d.line[i].position -= 0.12
        if (d.text == 'the' && i >= 4 && i <= 9) d.line[i].label += 1
        if (d.text == 'good' && i < 6)
          d.line[i].contri = -Math.abs(d.line[i].contri)
        else d.line[i].contri = Math.abs(d.line[i].contri)
        if (d.text == 'good' && i == 6) {
          d.line[i].contri = Math.abs(d.line[i + 1].contri)
        }
        if (d.text == 'good') d.line[i].contri *= 1.2
        if (d.text == 'n') d.line[i].contri = Math.abs(data[8].line[i].contri) * 1.2
        if (d.text == 'a' && i <= 5) {
          d.line[i].label = -1
        }
      }
      if (d.text == 'laughs') d.line[i].contri *= 0.8
      if (d.text == 'been' &&  d.line.length == 4) d.line[i].contri = Math.abs(d.line[i].contri)
      //d.line[i].contri = d.line[i].contri * 0.7 + d.line[i - 1].contri * 0.3
    }
    console.log(d.text, d.line.map(e => e.position))
  })

  const xScale = d3.scalePoint()
    .domain(data[0].line.map(d => d.layer))
    .rangeRound([0, width])

  const yScale = d3.scaleLinear()
    .domain([1, 0])  // Input
    .range([10, height]) // Output

  const getColorName = (x) => {
    if (x == positiveColor) {
      return 'pos'
    } else if (x == negativeColor) {
      return 'neg'
    } else if (x == neutralColor) {
      return 'neu'
    } else {
      return 'unknown'
    }
  }

  const addGradientColor = (defs) => {
    let colors = [
      [positiveColor, getColorName(positiveColor)],
      [negativeColor, getColorName(negativeColor)],
      ['#BBDEFB', 'neu'],
    ]

    let colors2 = [
      [positiveColor, getColorName(positiveColor)],
      [negativeColor, getColorName(negativeColor)],
      ['gray', 'neu'],
    ]
    for (let c1 of colors) {
      for (let c2 of colors) {
        let gradient = defs.append('linearGradient')
          .attr('id', `pattern-${c1[1]}-to-${c2[1]}`)
          .attr('x1', '0%')
          .attr('y1', '0%')
          .attr('x2', '100%')
          .attr('y2', '0%')

        gradient.append('stop')
          .attr('offset', '0%')
          .attr('stop-color', d3.interpolateRgb('white', c1[0])(0.2))

        gradient.append('stop')
          .attr('offset', '100%')
          .attr('stop-color', d3.interpolateRgb('white', c2[0])(0.2))
      }
    }
    for (let c1 of colors2) {
      for (let c2 of colors2) {
        for (let c3 of colors2) {
          let gradient = defs.append('linearGradient')
            .attr('id', `pattern-${c1[1]}-to-${c2[1]}-to-${c3[1]}-origin`)
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '100%')
            .attr('y2', '0%')

          gradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', d3.interpolateRgb('white', c1[0])(1))

          gradient.append('stop')
            .attr('offset', '50%')
            .attr('stop-color', d3.interpolateRgb('white', c2[0])(1))

          gradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', d3.interpolateRgb('white', c3[0])(1))
        }
      }
    }
  }

  const contriValues = [].concat(...raw.lines.map(d => d.line.map(e => Math.abs(e.contri)))).sort((a, b) => a - b)
  const contriThres = contriValues[~~(0.6 * contriValues.length)]
  const contriMax = contriValues[contriValues.length - 1]

  data.forEach(d => {
    for (let i = 1; i < d.line.length - 1; ++i) {
      let t1 = d.line[i - 1].contri
      let t2 = d.line[i + 1].contri
      if (Math.abs(t1) < contriThres && Math.abs(t2) < contriThres && t1 * t2 > 0) {
        d.line[i].contri = (t1 + t2) * 0.5
      }
    }
  })

  const contriValues2 = [].concat(...raw.lines.map(d => d.line.filter(e => e.display).map(e => Math.abs(e.contri)))).sort((a, b) => a - b)
  const contriThres2 = contriValues2[~~(0.66 * contriValues2.length)]

  const widthScale = (() => {
    let left = contriThres2
    let right = contriMax
    let scale1 = d3.scaleLinear().range([0.8, 1.0, 12]).domain([0, left, right])
    if (testmode) {
      scale1 = d3.scaleLinear().range([1.2, 1.5, 12]).domain([0, left, right])
    }
    return function (x) {
      x = Math.abs(x)
      // if (x < left) return 1
      // x = Math.min(9, scale1(x))
      return scale1(x)
    }
  })()

  const contriScale = (() => {
    let thres = contriThres
    let left = -thres
    let right = thres
    return function (x) {
      if (x <= left) {
        return positiveColor
      } else if (x >= right) {
        return negativeColor
      } else {
        return neutralColor
      }
    }
  })()

  const lineGenerator = (x1, x2, y1, y2, y3, y4, flag = 0) => {
    let m1 = x1 + (x2 - x1) * 0.5//r1
    let m2 = x1 + (x2 - x1) * 0.5//r2
    let x3 = x1 + (x2 - x1) * 0.15
    let x4 = x1 + (x2 - x1) * 0.85
    if (flag || Math.abs(y4 - y2) < 40) {
      x3 = x1
      x4 = x2
      m1 = x1 + (x2 - x1) * 0.5
      m2 = x1 + (x2 - x1) * 0.5
    }
    return `M${x1} ${y1}L${x1} ${y2} L${x3} ${y2} C${m1} ${y2} ${m2} ${y4} ${x4} ${y4} L${x2} ${y4} L${x2} ${y3} L${x4} ${y3} C${m2} ${y3} ${m1} ${y1} ${x3} ${y1} z`
  }

  const baselineGenerator = (x1, x2, y1, y2, y3, y4, r1 = 0.4, r2 = 0.6) => {
    let m1 = x1 + (x2 - x1) * r1
    let m2 = x1 + (x2 - x1) * r2
    y1 = y2 + 8
    y2 = y4 + 8
    if (Math.abs(y1 - y2) < 12) {
      let m = (y1 + y2) / 2
      y1 = y2 = m
    }
    return `M${x1 - 50} ${y1} L ${x1} ${y1}C${m1} ${y1} ${m2} ${y2} ${x2} ${y2} L ${x2 + 50} ${y2}`
  }

  async function LineChart() {
    const svg = d3.select("#network-svg")
    //.attr("width", width)
    //.attr("height", height)

    svg.selectAll("*").remove()
    const defs = svg.append('defs')
    const isalpha = val => /^[a-zA-Z]+$/.test(val)

    // Standard Margin Convention
    const background = svg.append("g")
      .attr('transform', `translate(${margin.left}, ${margin.top})`)
      .attr('class', 'background')

    // Call the x axis in a group tag
    const axis = background.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + (height - 10) + ")")
      .call(d3.axisBottom(xScale).tickFormat(d => d == 0 ? 'Input' : d))

    axis.select('.domain').remove()
    axis.selectAll('text')
      .attr('font-size', '16px')
      //.style('font-weight', 600)
      .style("font-family", "Arial, san-serif")

    axis.selectAll('line')
      .style('stroke', 'gray')
      .style('stroke-width', '2px')

    let shades = getShade(data)
    let extend_shades = getExtendShade(shades, data)

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
      d.path = lineGenerator(xScale(d.x), xScale(d.x + 1), d.y1, d.y2, d.y3, d.y4)
    })

    const shadeGroup = background
      .append('g')
      .attr('class', 'shade-group')

    const grayShadeElements = shadeGroup.selectAll('.gray-shade')
      .data(shades.filter(d => d.fill.indexOf('neg') == -1 && d.fill.indexOf('pos') == -1))
      .enter()
      .append('g')
      .attr('class', 'gray-shade')

    grayShadeElements
      .append('path')
      .attr('d', d => d.path)
      .style('stroke', 'none')
      .style('fill', d => d.fill)

    const coloredShadeElements = shadeGroup.selectAll('.colored-shade')
      .data(shades.filter(d => d.fill.indexOf('neg') != -1 || d.fill.indexOf('pos') != -1))
      .enter()
      .append('g')
      .attr('class', 'colored-shade')

    coloredShadeElements
      .append('path')
      .attr('d', d => d.path)
      .style('stroke', 'none')
      .style('fill', d => d.fill)

    const lineGroup = background
      .append('g')
      .attr('class', 'line-group')

    const lineElement = lineGroup.selectAll(".line")
      .data(data).enter()
      .append("g")
      .attr("class", "line")

    const glyphGroup = background
      .append('g')
      .attr('class', 'glyph-group')

    const labelGroup = background
      .append('g')
      .attr('class', 'label-group')
    /*
            labelGroup.append("text")
                .attr("x", -50)
                .attr("y", -8)
                .style("font-size", "13px")
                .style("font-family", "Roboto, san-serif")
                .text('word in sentence')
    */

    let labels = []
    for (let d of data) {
      labels.push({
        anchor: 'end',
        baseline: null,
        x0: 0,
        y0: yScale(d.line[0].position) + 5,
        fill: 'rgb(27, 30, 35)',
        s1: d.text.slice(0, d.pos),
        s2: d.text.slice(d.pos, d.pos + d.len),
        s3: d.text.slice(d.pos + d.len),
      })
    }

    const signSet = new Set(['.', '', ','])

    let phraseData = rawPhraseData
      .filter(d => d.type == 'phrase' && d.layer < n_rows)
      .filter(d => {
        let x = (xScale(d.layer - 1) * 3 + xScale(d.layer)) / 4
        let k = 0
        for (let i = d.top; i < d.bottom; ++i) {
          if (data[i].line[d.layer - 1].label != data[i + 1].line[d.layer - 1].label) {
            k = i
            break
          }
        }
        return (d.top != 0 && data[k].line[d.layer].display && data[k + 1].line[d.layer].display)
      })
      .filter(d => {
        return !signSet.has(data[d.top].text) && !signSet.has(data[d.bottom].text)
      })

    let relationData = rawPhraseData
      .filter(d => d.type == 'relation')
      .filter(d => d.layer < n_rows)
      .filter(d => data[d.top].line[d.layer - 1].display && data[d.bottom].line[d.layer].display)

    //if (testmode) relationData = []
    //console.log('relationData', relationData)
    relationData.forEach(d => d.removed = false)
    if (n_rows == 12) {
      for (let i = 0; i < relationData.length; ++i) if (!relationData[i].removed) {
        let t1 = relationData[i].top
        let b1 = relationData[i].bottom
        for (let j = i + 1; j < relationData.length; ++j) {
          let t2 = relationData[j].top
          let b2 = relationData[j].bottom
          if ((t1 == t2 && b1 == b2 || t1 == b2 && b1 == t2) && (Math.abs(relationData[i].layer - relationData[j].layer) <= 2)) {
            relationData[j].removed = true
          }
        }
      }
    }
    relationData = relationData
      .filter(d => !d.removed)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, n_cols * 1)

    let changeData = relationData
      .map(d => ({
        x: d.layer,
        y: d.bottom,
        left: data[d.bottom].line[d.layer - 1].contri,
        right: data[d.bottom].line[d.layer].contri,
      }))
      .filter(d => d.x < n_rows - 2)

    changeData = changeData.map(d => ({
      x: (xScale(d.x) + xScale(d.x - 1)) / 2,
      y: (yScale(data[d.y].line[d.x].position) + yScale(data[d.y].line[d.x - 1].position)) / 2,
      row: d.y,
      col: d.x,
      delta: Math.abs(d.left - d.right) * (d.left * d.right < 0 ? 2 : 1),
      left: d.left,
      right: d.right,
      removed: false,
    }))

    changeData = changeData
      .filter(d => contriScale(d.left) != contriScale(d.right))
      .sort((a, b) => b.delta - a.delta)
      .slice(0, 3)

    for (let i = 0; i < changeData.length; ++i) if (!changeData[i].removed) {
      for (let j = i + 1; j < changeData.length; ++j) {
        if (Math.abs(changeData[i].row - changeData[j].row) + Math.abs(changeData[i].col - changeData[j].col) <= 1) {
          changeData[j].removed = true
        }
      }
    }

    changeData = changeData.filter(d => !d.removed)

    changeData.forEach(d => {
      data[d.row].line[d.col - 1].has_rseg = 1
    })

    const phraseElement = glyphGroup.selectAll(".phrase")
      .data(phraseData)
      .enter()
      .append("g")
      .attr("class", "phrase")
      .style("display", "none")

    phraseElement.append('path')
      .attr('d', d => {
        let x = (xScale(d.layer - 1) * 3 + xScale(d.layer)) / 4
        let k = 0
        for (let i = d.top; i < d.bottom; ++i) {
          if (data[i].line[d.layer - 1].label != data[i + 1].line[d.layer - 1].label) {
            k = i
            break
          }
        }
        if (!data[d.top].line[d.layer - 1].display || !data[k + 1].line[d.layer - 1].display) return ''
        data[k].line[d.layer - 1].has_rseg = 1
        data[k + 1].line[d.layer - 1].has_rseg = 1
        let y = yScale((data[k].line[d.layer - 1].position + data[k + 1].line[d.layer - 1].position) / 2)
        let y1 = yScale(data[k].line[d.layer - 1].position)
        let y2 = yScale(data[k + 1].line[d.layer - 1].position)
        return `M${x - 20} ${y1} Q${x - 10} ${y - 2} ${x} ${y - 2} M${x - 20} ${y2} Q${x - 10} ${y + 2} ${x} ${y + 2}`
      })
      .style('fill', 'none')
      .style('stroke', 'rgb(27, 30, 35)')
      .style('opacity', .6)
      .style('stroke-width', '2px')

    phraseElement.append('path')
      .attr('transform', d => {
        let x = (xScale(d.layer - 1) * 3 + xScale(d.layer)) / 4
        let k = 0
        for (let i = d.top; i < d.bottom; ++i) {
          if (data[i].line[d.layer - 1].label != data[i + 1].line[d.layer - 1].label) {
            k = i
            break
          }
        }
        if (!data[d.top].line[d.layer - 1].display || !data[k + 1].line[d.layer - 1].display) return ''
        let y = yScale((data[k].line[d.layer - 1].position + data[k + 1].line[d.layer - 1].position) / 2)
        return `translate(${x + 3},${y})`
      })
      .attr('d', d => `M${1 * 6} ${0} L${-0.5 * 6} ${0.866 * 6} L${-0.5 * 6} ${-0.866 * 6} Z`)
      .style('fill', 'rgb(27, 30, 35)')
      .style('stroke', 'none')
      .style('opacity', .6)
      .style('stroke-width', '2px')

    const relationElement = shadeGroup.selectAll(".relation")
      .data(relationData)
      .enter()
      .append("g")
      .attr("class", "relation")

    relationElement
      .append("path")
      .attr("d", d => {
        let y1 = yScale(d.y1)
        let y2 = yScale(d.y2)
        if (testmode) {
          if (d.top == 2) d.top = 6
          if (d.top == 3) d.top = 4
          if (d.top == 6 && d.bottom == 9) d.layer -= 2
          y1 = yScale(data[d.top].line[d.layer - 1].position)
          y2 = yScale(data[d.bottom].line[d.layer].position)
        }
        let x0 = xScale(d.layer - 2)
        let x1 = xScale(d.layer - 1)
        let x2 = xScale(d.layer)
        let c1 = data[d.top].line[d.layer - 1].contri
        let c2 = data[d.top].line[d.layer].contri
        let c3 = data[d.bottom].line[d.layer - 1].contri
        let c4 = data[d.bottom].line[d.layer].contri
        let w1 = (widthScale(c1) + widthScale(c2)) / 4 - 0.5
        let w2 = (widthScale(c3) + widthScale(c4)) / 4 - 0.5
        let minw = Math.min(w1, w2)
        let x3 = (x0 + x1) / 2
        let x4 = (x1 + x2) / 2
        if (Math.abs(yScale(data[d.top].line[d.layer - 1].position) - yScale(data[d.top].line[d.layer - 2].position)) > 12) {
          x3 = x1
          y1 = yScale(data[d.top].line[d.layer - 1].position)
        }
        if (Math.abs(yScale(data[d.bottom].line[d.layer - 1].position) - yScale(data[d.bottom].line[d.layer].position)) > 12) {
          x4 = x2
          y2 = yScale(data[d.bottom].line[d.layer].position)
        }
        w1 = Math.min(w1, minw * 3)
        w2 = Math.min(w2, minw * 3)
        let path = lineGenerator(x3, x4, y1 - w1, y1 + w1, y2 - w2, y2 + w2, 1)
        // console.log('relation', d.top, d.bottom, data[d.top].line[d.layer].label, data[d.bottom].line[d.layer].label)
        return path
      })
      .style("stroke-width", testmode ? 2 : .5)
      // .style("stroke-dasharray", "8,8")
      .style("stroke", d => {
        let c1 = data[d.top].line[d.layer - 2].contri
        let c2 = data[d.top].line[d.layer - 1].contri
        let c3 = data[d.bottom].line[d.layer].contri
        c1 = getColorName(contriScale(c1))
        c2 = getColorName(contriScale(c2))
        c3 = getColorName(contriScale(c3))
        let fill = `url(#pattern-${c1}-to-${c2}-to-${c3}-origin)`
        return fill
      })
      .style("fill", d => {
        let c1 = data[d.top].line[d.layer - 2].contri
        let c2 = data[d.top].line[d.layer - 1].contri
        let c3 = data[d.bottom].line[d.layer].contri
        c1 = getColorName(contriScale(c1))
        c2 = getColorName(contriScale(c2))
        c3 = getColorName(contriScale(c3))
        let fill = `url(#pattern-${c1}-to-${c2}-to-${c3}-origin)`
        return fill
      })

    let changeElement = glyphGroup.selectAll('.represent')
      .data(changeData).enter()
      .append('g')
      .attr('class', 'represent')
      .attr('transform', d => `translate(${d.x},${d.y})`)

    if (testmode) {
      changeElement.append('rect')
        .attr('x', -12)
        .attr('y', d => -6.5)
        .attr('width', 24)
        .attr('height', d => 13)
        .attr('rx', 2)
        .attr('ry', 2)
        .style('stroke', d => contriScale(d.right))
        .style('stroke-width', 1)
        .style('fill', d => contriScale(d.right))

      changeElement.append('rect')
        .attr('x', -12)
        .attr('y', d => -6.5)
        .attr('width', 12)
        .attr('height', d => 13)
        .attr('rx', 2)
        .attr('ry', 2)
        .style('fill', d => contriScale(d.left))

    } else {
      changeElement.append('rect')
        .attr('x', -9)
        .attr('y', d => -5)
        .attr('width', 18)
        .attr('height', d => 10)
        .attr('rx', 2)
        .attr('ry', 2)
        .style('stroke', d => contriScale(d.right))
        .style('stroke-width', 1)
        .style('fill', d => contriScale(d.right))

      changeElement.append('rect')
        .attr('x', -9)
        .attr('y', d => -5)
        .attr('width', 9)
        .attr('height', d => 10)
        .attr('rx', 2)
        .attr('ry', 2)
        .style('fill', d => contriScale(d.left))

    }
    addGradientColor(defs)

    let segments = []
    data.forEach(function (d, index) {
      for (let i = 0; i + 1 < d.line.length; ++i) {
        d.line[i].has_shade = false
        let w1 = widthScale(d.line[i].contri)
        let w2 = widthScale(d.line[i + 1].contri)
        if (!d.line[i].display) {
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
          let tw = measure(d.text)[0]
          let t = Math.min(y1, y3)
          let b = Math.max(y1 + w1, y3 + w2)
          let x3 = Math.max(x1, (x1 + x2) / 2 - tw * (testmode ? (d.text == 'the' ? 0.15 : 0.45) : 0.48))// - delta)
          let x4 = Math.min(x2, (x1 + x2) / 2 + tw * (testmode ? (d.text == 'the' ? 0.15 : 0.45) : 0.48))// + delta)
          defs.append('clipPath')
            .attr('id', `clipping-${d.line[i].fill_id}`)
            .append('path')
            .attr('d', `M${x1} ${t} L${x1} ${b} L${x3} ${b} L${x3} ${t} M${x4} ${t} L${x4} ${b} L${x2} ${b} L${x2} ${t}`)

          /*
          svg.append('defs')
            .append('path')
            .attr('id', `labelbaseline-${d.line[i].fill_id}`)
            .attr('d', path)
            */
        }

        segments.push({
          d: lineGenerator(x1, x2, y1, y1 + w1, y3, y3 + w2),
          opacity: 1,
          fill_id: d.line[i].fill_id,
          center: { x: (x1 + x2) / 2, y: (y1 + y3) / 2 },
          baseline: d.line[i].show_label && isalpha(d.text) ? [x1, x2, y1, y1 + w1, y3, y3 + w2] : null,
          label: d.line[i].show_label && !d.line[i].has_rseg ? { text: d.text, pos: d.pos, len: d.len } : null,
          x: i,
          y: index,
        })
      }
    })

    const segElement = lineGroup
      .selectAll('.segment')
      .data(segments).enter()
      .append('g')
      .attr('class', 'segment')

    const segLine = segElement
      .append('path')
      .attr('d', d => d.d)
      .style('clip-path', d => d.label ? `url(#clipping-${d.fill_id})` : 'none')
      .style('fill', d => `url(#pattern-${d.fill_id})`)
      .style('stroke', 'none')
      .style('opacity', d => d.opacity)

    for (let d of segments) {
      if (d.baseline == null || d.label == null) continue
      labels.push({
        anchor: 'middle',
        x0: d.center.x,
        y0: d.center.y,
        fill: 'rgb(27, 30, 35)',
        baseline: d.baseline,
        s1: d.label.text.slice(0, d.label.pos),
        s2: d.label.text.slice(d.label.pos, d.label.pos + d.label.len),
        s3: d.label.text.slice(d.label.pos + d.label.len),
      })
    }

    for (let d of data) {
      labels.push({
        anchor: 'start',
        is_end: d.end_point == d.line.length - 1 ? 1 : 0,
        contri: Math.abs(d.line[d.end_point].contri),
        x0: xScale(d.end_point) + 8,
        y0: yScale(d.line[d.end_point].position) + 3,
        baseline: null,
        fill: d.end_point + 1 < d.line.length ? contriScale(d.line[d.end_point].contri) : 'rgb(27, 30, 35)',
        s1: d.text.slice(0, d.pos),
        s2: d.text.slice(d.pos, d.pos + d.len),
        s3: d.text.slice(d.pos + d.len),
      })
    }
    const endLabelGroup = labelGroup
      .selectAll('.end-label')
      .data(data).enter()
      .append('g')
      .attr('class', 'end-label')

    const endGlyph = endLabelGroup.filter(d => !d.line[n_rows - 1].display)

    endGlyph
      .append('circle')
      .attr('cx', d => xScale(d.end_point))
      .attr('cy', d => yScale(d.line[d.end_point].position))
      .attr('r', 3)
      .style('fill', d => contriScale(d.line[d.end_point].contri))//'#ccc')
      .style('stroke', d => contriScale(d.line[d.end_point].contri))
      .style('stroke-width', 1.5)

    labels.forEach(d => {
      d.all_pieces = [[d.s1, 0.4], [d.s2, 1], [d.s3, 0.4]].filter(e => e[0].length > 0)
      d.pieces = []
      let shortcut = ''
      for (let e of d.all_pieces) {
        if (e[0].length + shortcut.length > max_text_len) {
          const curr = e[0].slice(0, max_text_len - shortcut.length) + '...'
          d.pieces.push([curr, e[1]])
          shortcut += curr
          break
        } else {
          d.pieces.push(e)
          shortcut += e[0]
        }
      }
      const m = measure(shortcut)
      d.width = m[0]
      d.height = m[1] * 0.7
      if (d.anchor == 'end') {
        d.x0 -= d.width
        d.anchor = 'start'
      } else if (d.anchor == 'middle') {
        d.x0 -= d.width * 0.5
        d.y0 += d.height// * 0.5
      }
      d.x = d.x0
      d.y = d.y0
    })


    let endLabels = labels
      .filter(d => d.is_end)
      .sort((a, b) => b.contri - a.contri)
      .slice(0, 10)
    labels = labels.filter(d => !d.is_end)
    labels = labels.concat(endLabels)
    const simulation = d3.forceSimulation(labels)
      .force("x", d3.forceX(0).strength(0.0))
      .force("y", d3.forceY(0).strength(0.0))
      .force("collide", forceCollide())

    for (let i = 0; i < 50; ++i) {
      await simulation.tick()
    }

    // paint labels with
    const allLabel = background.selectAll('.label')
      .data(labels).enter()
      .append('text')
      .attr('class', 'label')
      .attr('fill', d => d.fill)
      .attr('text-anchor', d => d.anchor)
      .style('font-size', label_size)
      .style('font-family', label_font)

    let labelWithPath = allLabel.filter(d => d.baseline)
    let labelNoPath = allLabel.filter(d => !d.baseline)

    labelWithPath
      .append('defs').append('path')
      .attr('id', (d, i) => `labelbaseline-${i}`)
      .attr('d', d => {
        let dx = d.x - d.x0
        let dy = d.y - d.y0
        let v = d.baseline
        return baselineGenerator(v[0] + dx, v[1] + dx, v[2] + dy, v[3] + dy, v[4] + dy, v[5] + dy)
      })

    let labelText = labelWithPath
      .append('textPath')
      .attr('href', (d, i) => `#labelbaseline-${i}`)
      .attr('startOffset', '50%')

    labelText.selectAll("tspan")
      .data(d => d.pieces).enter()
      .append("tspan")
      .style("opacity", d => d[1])
      .text(d => d[0])

    let labelText2 = labelNoPath
      .attr('x', d => d.x)
      .attr('y', d => d.y)

    labelText2.selectAll("tspan")
      .data(d => d.pieces).enter()
      .append("tspan")
      .style("opacity", d => d[1])
      .text(d => d[0])

    d3.select("body").selectAll(".svg-tooltip").remove()

    const tooltip = d3.select("body").append("div")
      .attr("class", "svg-tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")

    const update_tooltip = (d) => {
      tooltip.selectAll("span")
        .data(d.all_pieces).enter()
        .append("span")
        .text(e => e[0])

      tooltip.selectAll("span")
        .data(d.all_pieces)
        .style("opacity", e => e[1])
        .text(e => e[0] + ' ')

      tooltip.selectAll("span")
        .data(d.all_pieces)
        .exit().remove()

    }

    labelWithPath
      .on("mouseover", function(d){
        update_tooltip(d)
        return tooltip.style("visibility", "visible");
      })
      .on("mousemove", function(){
        return tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px");
      })
      .on("mouseout", function(){
        return tooltip.style("visibility", "hidden");
      })

    labelNoPath
      .on("mouseover", function(d){
        update_tooltip(d)
        return tooltip.style("visibility", "visible");
      })
      .on("mousemove", function(){
        return tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px");
      })
      .on("mouseout", function(){
        return tooltip.style("visibility", "hidden");
      })
  }

  LineChart()
}

export default paintSampleComposition
