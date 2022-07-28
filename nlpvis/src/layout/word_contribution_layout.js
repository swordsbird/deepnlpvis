
import * as d3 from "d3"
import wordcloud from "../plugins/wordcloud"
import { stepSizeScale, getValuesByEntropy } from "../plugins/utils"
import bus from "../plugins/bus"

export async function paintLineChart(self, highlight_word = null, font_size) {
    const sleep = (timeout) =>
      new Promise((resolve) => {
        setTimeout(resolve, timeout)
      })
  
    let linechart = d3.select("#linechart-g")
    let n = self.layers.length
    let padding = self.linechart.padding + 5
    let width = self.linechart.width - padding * 2
    let height = self.linechart.height - padding - 25
    let chartline = null
    window.layers = self.layers
  
    if (linechart.select(".chartline").empty()) {
      chartline = linechart
        .append("g")
        .attr("class", "chartline")
        .attr("transform", `translate(${padding},${0})`);
    } else {
      chartline = linechart.select(".chartline");
    }
  
    linechart = chartline;
    linechart.selectAll("*").remove();
  
    // let xScale = d3.scaleLinear().domain([1, n]).range([0, width])
    let xScale = d3
      .scaleLinear()
      .range([0, width])
      .domain(d3.extent(self.layers, (d) => d.index));
    let entropy_range = [0, 1];
    let yScale = d3.scaleLinear().domain(entropy_range).range([height, 0]);
  
    let line = d3
      .line()
      .x(function (d, i) {
        return xScale(d.index);
      })
      .y(function (d) {
        return yScale(d.info);
      })
      .curve(d3.curveMonotoneX);
  
    // 3. Call the x axis in a group tag
    let xAxis = linechart
      .append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(
        d3
          .axisBottom(xScale)
          .tickValues(self.layers.map((d, i) => i))
          .tickFormat((d) => ~~d)
      );
    xAxis.selectAll("text").attr("dx", -(xScale(1) - xScale(0)) * 0.5); // Create an axis component with d3.axisBottom
  
    xAxis
      .append("text")
      .attr("x", width + 5)
      .attr("y", 20)
      .attr("text-anchor", "end")
      .attr("fill", "#000")
      .attr("font-size", "16px")
      .text("Layer");
  
    xAxis
      .selectAll("text")
      .attr("font-size", "16px")
      //.style('font-weight', 600)
      .style("font-family", "Roboto, san-serif");
  
    let yAxis = linechart
      .append("g")
      .attr("class", "y axis")
      .call(
        d3.axisLeft(yScale).tickFormat((d) => Math.floor(d * 100 + 1e-3) + "%")
      ); // Create an axis component with d3.axisLeft
  
    yAxis
      .append("text")
      .attr("x", -20)
      .attr("y", -7)
      .attr("text-anchor", "start")
      .attr("fill", "#000")
      .attr("font-size", "16px")
      .text("Word Contribution Percentile")
      .style("font-family", "sans-serif");
    if (self.labels.length == 4)
      self.layers.forEach((d, i) => {
        if (i < 9) d.info = 0;
      });
  
    yAxis
      .selectAll("text")
      .attr("font-size", "16px")
      //.style('font-weight', 600)
      .style("font-family", "Roboto, san-serif");
  
    linechart
      .append("path")
      .attr("class", "line") // Assign a class for styling
      .attr("d", line(self.layers))
      .style("fill", "none")
      .style("opacity", 0.6)
      .style("stroke", (d, i) => "gray")
      .style("stroke-width", 3);
  
    let dots = linechart
      .selectAll(".dot")
      .data(self.layers)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("cx", function (d, i) {
        return xScale(d.index);
      })
      .attr("cy", function (d) {
        return yScale(d.info);
      })
      .attr("r", 3)
      .style("fill", (d) => "gray")
      .style("opacity", 0.5);
  
    let all_words = [];
  
    let highlight_set = new Set();
    if (highlight_word) {
      for (let i = 0; i < highlight_word.retained_words.length; ++i) {
        if (!highlight_word.retained_words[i]) continue;
        for (let d of highlight_word.retained_words[i]) {
          highlight_set.add(d[0]);
        }
      }
      for (let i = 0; i < highlight_word.discarded_words.length; ++i) {
        if (!highlight_word.discarded_words[i]) continue;
        for (let d of highlight_word.discarded_words[i]) {
          highlight_set.add(d[0]);
        }
      }
    }
  
    let important_words = new Set();
    for (let i = 0; i < self.layers.length; ++i) {
      if (self.layers[i].words) {
        important_words = new Set(
          self.layers[i].words
            .sort((a, b) => b.value - a.value)
            .slice(0, 3)
            .map((d) => d.word)
        );
      }
    }
  
    let first_words = null,
      last_words = null;
    let word_counts = {};
    for (let i = 0; i < self.layers.length; ++i) {
      if (self.layers[i].words) {
        getValuesByEntropy(self.layers[i].words, self, i);
        let layer = self.layers[i];
        let words = layer.words.sort((a, b) => b.value - a.value);
        if (!first_words) first_words = words;
        last_words = words;
        let max_size = font_size[1]
        /*
        let best_power = 2,
          min_delta = 1e8;
  
        for (let power = 0.25; power <= 5; power += 0.05) {
          const sizeScale = d3
            .scalePow()
            .exponent(power)
            .domain(global_value_range)
            .range(global_font_size_range);
          let current = words
            .map((d) => Math.min(sizeScale(d.value), max_size) / max_size)
            .filter((d) => !isNaN(d));
          let target = last_font_size_distribution;
          let delta = 0;
          for (let i = 0; i < current.length && i < target.length; ++i) {
            delta += (current[i] - target[i]) * (current[i] - target[i]);
          }
          // break
          if (delta < min_delta) {
            min_delta = delta;
            best_power = power;
          }
        }
        const sizeScale = d3
          .scalePow()
          .exponent(best_power)
          .domain(global_value_range)
          .range(global_font_size_range)
        */
        let sizeScale;
        if (self.n_layer == 4) {
          sizeScale = stepSizeScale(
            font_size,
            words.map((d) => d.value),
            4,
            [4, 7, 12]
          );
        } else {
          sizeScale = stepSizeScale(
            font_size,
            words.map((d) => d.value),
            4,
            [1, 4, 8]
          );
        }
  
        words.forEach((d) => {
          d.size = Math.min(sizeScale(d.value), max_size);
        });
  
        // console.log('best power', i, best_power, min_delta)
  
        let last = layer.word_range[0];
        let next = layer.word_range[1];
        let x0 = xScale(last);
        let x1 = xScale(next);
        if (next - last >= 1) {
          x0 += 9;
          x1 -= 9;
        }
  
        words = words.sort((a, b) => {
          if (highlight_set.has(a.word)) {
            return 1;
          } else if (highlight_set.has(b.word)) {
            return -1;
          }
          return b.value - a.value;
        });
  
        for (let d of words) {
          d.stroke = "none";
          d.stroke_width = 1.5;
          if (d.status != "retained") {
            if (d.status == "old_discarded") {
              d.fill = self.neutralColorGray //"lightgray";
            } else {
              d.fill = self.neutralColorGray
            }
          } else {
            d.fill = self.getWordColor(d);
          }
          d.highlight = highlight_set.has(d.word);
          // d.opacity = d.status == 'old_discarded' ? 0.5 : 1
        }
  
        let barriers = [];
        let delta_x = (x1 - x0) / 20;
        let delta_y = (next - last) / 20;
        for (let k = 0; k < 20; ++k) {
          let y0 = last + delta_y * k,
            y1;
          let d = y0 % 1;
          y0 -= d;
          y1 = y0 + 1;
          // console.log(y0, y1)
          y0 = yScale(self.layers[y0].info);
          y1 = yScale(self.layers[y1].info);
          y0 = y0 * (1 - d) + y1 * d;
          barriers.push({
            x0: delta_x * k,
            x1: delta_x * (k + 1),
            y0: y0 - 10,
            y1: height,
          });
        }
        let retained_words = words
          .filter((d) => d.status == "retained" && d.word.length > 1)
          .sort((a, b) => b.size - a.size)
        let topk = 5;
        for (let k = 0; k < topk; ++k) {
          retained_words[k].ex = 16;
        }
        for (let k = topk; k < retained_words.length; ++k) {
          retained_words[k].ex = 0;
        }
  
        // console.log(retained_words.map((d) => d.word).slice(0, 10));
        await wordcloud()
          .size([x1 - x0, height])
          .data(retained_words)
          .text((d) => d.word)
          .font("Roboto")
          .padding(6)
          .keepFirst(
            (d) =>
              important_words.has(d.word) || (i == 11 && d.word == "recommend")
          )
          .expandx((d) => d.ex)
          .fontSize((d) => d.size)
          .fontStyle((d) => (d.status == "retained" ? "normal" : "italic"))
          .fontWeight((d) => "bold")
          .rangey((d) => [
            Math.max(0, 1 - d.yaxis - 0.2),
            Math.min(
              1,
              next == xScale.domain()[1] ? 1 - d.yaxis + 0.35 : 1 - d.yaxis + 0.25
            ),
            1 - d.yaxis,
          ])
          .barriers(barriers)
          .start((words) => {
            // console.log(words.slice(0, 5))
            let max_words = 50,
              max_words_per_line = 6,
              n_words = 0,
              count_words = {};
  
            if (self.n_layer == 4) {
              max_words_per_line = 75;
              max_words_per_line = 8;
            }
            console.log(i, words.filter(d => d.expandx > 0)) 
            words = words
              // .filter((d) => d.display)
              .map((d) => ({
                x: d.x + x0 + 8,
                y: d.y,
                display: d.display,
                size: d.size,
                text: d.text,
                width: d.width,
                height: d.height,
                layer: i,
                style: d.style,
                weight: d.weight,
                id: 0,
                highlight: d.highlight,
                fill: d.fill,
                stroke: d.stroke,
                contri: d.contri,
                glyph: d.expandx > 0 && d.display,
                stroke_width: d.stroke_width,
                opacity: d.opacity,
              }));
  
            for (let d of words)
              if (d.display) {
                let y = ~~((d.y / height) * 10);
                count_words[y] = count_words[y] || 0;
                if (count_words[y] >= max_words_per_line) {
                  d.display = 0;
                  continue;
                }
                count_words[y]++;
              }
  
            for (let d of words)
              if (d.display) {
                if (n_words >= max_words) {
                  d.display = 0;
                  continue;
                }
                n_words++;
                word_counts[d.text] = (word_counts[d.text] || 0) + 1;
              }
            // .slice(0, 100);
            all_words = all_words.concat(words);
          });
  
        barriers = [];
        for (let k = 0; k < 20; ++k) {
          let y0 = last + delta_y * k,
            y1;
          let d = y0 % 1;
          y0 -= d;
          y1 = y0 + 1;
          y0 = yScale(self.layers[y0].info);
          y1 = yScale(self.layers[y1].info);
          y0 = y0 * (1 - d) + y1 * d;
          barriers.push({
            x0: delta_x * k,
            x1: delta_x * (k + 1),
            y0: 0,
            y1: y0 + 15,
          });
        }
  
        let discarded_words = words.filter((d) => d.status != "retained" && d.word != "good");
        await wordcloud()
          .size([x1 - x0, height])
          .data(discarded_words)
          .text((d) => d.word)
          .font("Roboto")
          .padding(8)
          .fontSize((d) => d.size)
          .fontStyle((d) => (d.status == "retained" ? "normal" : "italic"))
          .fontWeight((d) => "bold")
          .rangey((d) => [
            Math.max(0, 1 - d.yaxis - 0.1),
            Math.min(1, 1 - d.yaxis + 0.3),
            1 - d.yaxis,
          ])
          .barriers(barriers)
          .start((words) => {
            words = words
              // .filter((d) => d.display)
              .map((d) => ({
                x: d.x + x0 + 8,
                y: d.y,
                size: d.size,
                text: d.text,
                width: d.width,
                height: d.height,
                layer: i,
                contri: d.contri,
                style: d.style,
                display: d.display,
                weight: d.weight,
                id: 1,
                highlight: d.highlight,
                fill: d.fill,
                stroke: d.stroke,
                stroke_width: d.stroke_width,
                opacity: d.opacity,
              }));
            let max_words = 75,
              max_words_per_line = 8,
              n_words = 0,
              count_words = {};
  
            for (let d of words)
              if (d.display) {
                let y = ~~((d.y / height) * 10);
                count_words[y] = count_words[y] || 0;
                if (count_words[y] >= max_words_per_line) {
                  d.display = 0;
                  continue;
                }
                count_words[y]++;
              }
  
            for (let d of words)
              if (d.display) {
                if (n_words >= max_words) {
                  d.display = 0;
                  continue;
                }
                n_words++;
                word_counts[d.text] = (word_counts[d.text] || 0) + 1;
              }
            //.slice(0, 50);
            all_words = all_words.concat(words);
          });
      }
    }
  
    let down_words = first_words
      .filter((d) => d.display)
      .sort((a, b) => b.value - a.value)
      .slice(0, 20);
  
    // for (let x of down_words) console.log('down', x.word, x.line)
    down_words = down_words.filter(
      (d) =>
        (d.line[0] > d.line[d.line.length - 1] + 5 &&
          d.line[0] >= d.line[d.line.length - 2] &&
          d.line[d.line.length - 2] >= d.line[d.line.length - 1] &&
          d.line[1] > 40 &&
          d.line[d.line.length - 1] <= 65 &&
          word_counts[d.word] >= 3) ||
        d.word == "film"
    );
  
    down_words = down_words.filter((d) => d.word.length > 2).slice(0, 2);
  
    down_words = down_words.map((d) => d.word);
  
    let down_word_lines = down_words
      .map((d) =>
        [].concat(
          ...all_words
            .filter((e) => e.text == d)
            .map((e) => [
              { x: e.x, y: e.y - e.size / 4 },
              { x: e.x + e.width, y: e.y - e.size / 4 },
            ])
        )
      )
      .map((d) =>
        [].concat([{ x: 50, y: d[0].y }], d, [
          { x: width + 15, y: d[d.length - 1].y },
        ])
      );
  
    let up_words = last_words
      .filter((d) => d.display)
      .sort((a, b) => b.value - a.value)
      .slice(0, 20)
      .filter(
        (d) =>
          d.line[0] < d.line[d.line.length - 1] - 10 &&
          d.line[0] <= d.line[d.line.length - 2] &&
          d.line[d.line.length - 2] <= d.line[d.line.length - 1] &&
          d.line[1] > 40 &&
          d.line[d.line.length - 1] >= 60 &&
          word_counts[d.word] >= 3
      );
  
    up_words = up_words
      .filter((d) => d.word.length > 2)
      .slice(0, 2)
      .map((d) => d.word);
  
    let up_word_lines = up_words
      .map((d) =>
        [].concat(
          ...all_words
            .filter((e) => e.text == d)
            .map((e) => [
              { x: e.x, y: e.y - e.size / 4 },
              { x: e.x + e.width, y: e.y - e.size / 4 },
            ])
        )
      )
      .map((d) =>
        [].concat([{ x: 50, y: d[0].y }], d, [
          { x: width + 15, y: d[d.length - 1].y },
        ])
      );
  
    let down_word_underlines = down_words
      .map((d) =>
        [].concat(
          ...all_words
            .filter((e) => e.text == d && e.display)
            .map((e) => [
              {
                x1: e.x + 20,
                y1: e.y + e.size / 4,
                x2: e.x + e.width + 20,
                y2: e.y + e.size / 4,
              },
            ])
        )
      )
      .map((d) =>
        [].concat([{ x: 50, y: d[0].y }], d, [
          { x: width + 15, y: d[d.length - 1].y },
        ])
      );
  
    let up_word_underlines = up_words
      .map((d) =>
        [].concat(
          ...all_words
            .filter((e) => e.text == d && e.display)
            .map((e) => [
              {
                x1: e.x + 20,
                y1: e.y + e.size / 4,
                x2: e.x + e.width + 20,
                y2: e.y + e.size / 4,
              },
            ])
        )
      )
      .map((d) =>
        [].concat([{ x: 50, y: d[0].y }], d, [
          { x: width + 15, y: d[d.length - 1].y },
        ])
      );
  
    let highlight_line = d3
      .line()
      .x(function (d, i) {
        return d.x;
      })
      .y(function (d) {
        return d.y;
      })
      .curve(d3.curveMonotoneX);
  
    up_word_lines = up_word_lines.map((d) => highlight_line(d));
    down_word_lines = down_word_lines.map((d) => highlight_line(d));
  
    self.trending_up_lines = up_word_lines;
    self.trending_down_lines = down_word_lines;
    self.trending_up_underlines = up_word_underlines;
    self.trending_down_underlines = down_word_underlines;
    /*
    function wordInteraction(word) {
      let hover_word_item = null;
      return word
        .on("click", async (d) => {
          if (self.enable_word_filter) {
            let resp = self.word_cache[d.text];
            let idxs;
            if (resp) {
              idxs = resp.data.idxs;
            } else {
              idxs = self.sentences
                .filter((e, i) => e.text.indexOf(d.text) != -1)
                .map((e) => e.index);
            }
            let oldidxs = new Set(self.current_items.map((e) => e.index));
            idxs = idxs.filter((e) => !oldidxs.has(e));
            resp = await self.getSamples(idxs);
            self.current_items = resp
              .concat(self.current_items)
              .sort(
                (a, b) =>
                  (a.text.indexOf(d.text) == -1) - (b.text.indexOf(d.text) == -1)
              );
          } else {
            let resp = self.word_cache[d.text];
            let idxs;
            if (resp) {
              idxs = resp.data.idxs;
            } else {
              idxs = self.sentences
                .filter((e, i) => e.text.indexOf(d.text) != -1)
                .map((e) => e.index);
            }
            let oldidxs = new Set(self.current_items.map((e) => e.index));
            idxs = idxs.filter((e) => !oldidxs.has(e));
            if (idxs.length >= 10) self.addWordDAG(d.text);
          }
        })
        .on("mouseover", (d) => {
          if (
            !isalpha(d.text) ||
            d.text.length <= 1 ||
            d.text.indexOf("_") != -1
          ) {
            return;
          }
          hover_word_item = d.text;
          let left = d3.event.pageX + 10;
          let top = d3.event.pageY - 10;
          const show_detail = async () => {
            let resp = null;
            if (self.word_cache[d.text]) {
              resp = self.word_cache[d.text];
            } else {
              resp = await axios.post(`${this.server_url}/api/word`, {
                word: d.text,
              });
              self.word_cache[d.text] = resp;
            }
            let idxs = resp.data.idxs;
  
            if (self.enable_word_filter) {
            } else {
              let info = resp.data.info;
              for (let i = 1; i < info.length; ++i) {
                info[i] = Math.min(info[i - 1], info[i]);
              }
              info = info.map((e) => ({ info: e }));
  
              linechart
                .append("path")
                .attr("class", "exline") // Assign a class for styling
                .attr("d", line(info))
                .style("fill", "none")
                .style("opacity", 0.6)
                .style("stroke", d.fill)
                .style("stroke-width", 3);
  
              linechart
                .selectAll(".exdot")
                .data(info)
                .enter()
                .append("circle")
                .attr("class", "exdot")
                .attr("cx", function (d, i) {
                  return xScale(i + 1);
                })
                .attr("cy", function (d) {
                  return yScale(d.info);
                })
                .attr("r", 3)
                .style("fill", d.fill)
                .style("opacity", 0.5);
  
              chartword.style("opacity", 0.5);
            }
  
            self.highlightGrid(idxs);
  
            if (self.enable_word_filter) {
            } else {
              let text = `<p>${d.text}, ${idxs.length} samples </p>`;
  
              resp = await self.getSamples(idxs);
              text =
                text +
                resp
                  .slice(0, 8)
                  .map((e, index) => `#${index + 1}: ${e.text}`)
                  .join("</br>");
              if (hover_word_item == d.text) {
                self.showTooltip({ top, left, content: text });
              }
            }
          };
          setTimeout(() => {
            if (hover_word_item == d.text) {
              show_detail();
            }
          }, 500);
        })
        .on("mouseout", (d) => {
          hover_word_item = null;
          self.hideTooltip();
          self.highlightGrid(null);
          linechart.selectAll(".exline").remove();
          linechart.selectAll(".exdot").remove();
          chartword.style("opacity", 1);
        });
    }*/
  
    self.infolossWords = all_words
  }
  