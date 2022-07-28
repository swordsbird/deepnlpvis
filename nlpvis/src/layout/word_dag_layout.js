
import * as d3 from "d3"
import bus from "../plugins/bus"
import wordcloud from "../plugins/wordcloud"
import { stepSizeScale } from "../plugins/utils"

export function layoutDAGEdges(state, layers) {
  for (let i = 0; i < layers.length; ++i) {
    layers[i].clusters.forEach(d => {
      if (d.in_edges && d.in_edges.length > 0) {
        let tot_weight = 0
        let height_threshold = 12
        let weight_threshold = 0
        let weights = []
        for (let e of d.in_edges) weights.push(state.edges[e].weight)
        let n = weights.length - 1
        weights.sort((a, b) => b - a)

        for (let e of d.in_edges) tot_weight += state.edges[e].weight
        weight_threshold = height_threshold / d.height * tot_weight

        for (let e of d.in_edges) {
          if (state.edges[e].weight < weight_threshold && !state.edges[e].is_straight) {
            tot_weight -= state.edges[e].weight
            state.edges[e].weight = 0
          }
        }
        tot_weight = 0
        for (let e of d.in_edges) tot_weight += state.edges[e].weight

        let curr_weight = 0
        for (let e of d.in_edges) {
          state.edges[e].x2 = d.x
          state.edges[e].y3 = d.y + d.height * (curr_weight / tot_weight)
          curr_weight += state.edges[e].weight
          state.edges[e].y4 = d.y + d.height * (curr_weight / tot_weight)
        }
      }
    })
  }
  for (let i = 0; i < layers.length; ++i) {
    layers[i].clusters.forEach(d => {
      if (d.out_edges && d.out_edges.length > 0) {
        for (let j = 0; j < d.out_edges.length; ++j) {
          let e = d.out_edges[j]
          state.edges[e].x1 = d.x + d.width
          state.edges[e].y1 = j == 0 ? d.y : (state.edges[d.out_edges[j - 1]].y2)
          state.edges[e].y2 = state.edges[e].y1 + (state.edges[e].y4 - state.edges[e].y3)
        }
        let last = d.out_edges[d.out_edges.length - 1]
        if (state.edges[last].y2 > d.y + d.height) {
          let scale = d.height / (state.edges[last].y2 - d.y)
          for (let j = 0; j < d.out_edges.length; ++j) {
            let e = d.out_edges[j]
            state.edges[e].y1 = (state.edges[e].y1 - d.y) * scale + d.y
            state.edges[e].y2 = (state.edges[e].y2 - d.y) * scale + d.y
            state.edges[e].y4 = (state.edges[e].y4 - state.edges[e].y3) * scale + state.edges[e].y3
          }
        }
      }
    })
  }
  for (let i = 0; i < layers.length; ++i) {
    layers[i].clusters.forEach(d => {
      for (let j = 0; j < d.in_edges.length; ++j) {
        let e = d.in_edges[j]
        let h = state.edges[e].y4 - state.edges[e].y3
        state.edges[e].y3 = j == 0 ? d.y : (state.edges[d.in_edges[j - 1]].y4)
        state.edges[e].y4 = state.edges[e].y3 + h
      }
    })
  }
  for (let i = 0; i < layers.length; ++i) {
    layers[i].clusters.forEach(d => {
      let last = d.in_edges.length - 1
      if (last < 0) return
      let e = d.in_edges[last]
      if (state.edges[e].y4 < d.y + d.height) {
        let delta = d.y + d.height - state.edges[e].y4
        delta = delta / 2
        for (let j = 0; j < d.in_edges.length; ++j) {
          let e = d.in_edges[j]
          state.edges[e].y3 += delta
          state.edges[e].y4 += delta
        }
      }
    })
    layers[i].clusters.forEach(d => {
      let last = d.out_edges.length - 1
      if (last < 0) return
      let e = d.out_edges[last]
      if (state.edges[e].y2 < d.y + d.height) {
        let delta = d.y + d.height - state.edges[e].y2
        delta = delta / 2
        for (let j = 0; j < d.out_edges.length; ++j) {
          let e = d.out_edges[j]
          state.edges[e].y1 += delta
          state.edges[e].y2 += delta
        }
      }
    })
  }
}

export async function paintWordDAG(self) {
  let DAG = d3.select("#network-svg")
  DAG.selectAll("*").remove();

  let svg = DAG.selectAll("neuron-cluster")
    .data(self.neuron_clusters)
    .enter()
    .append("g")
    .attr("class", "neuron-cluster")
    .attr("transform", (d) => `translate(${d.x},${d.y})`);

  let timer = null
  const on_mouseover = function (d) {
    let left = d3.event.pageX + 10
    let top = d3.event.pageY - 10
    if (!d.idxs) return
    clearTimeout(timer)
    timer = setTimeout(() => {
      d3.select(this)
        .style("stroke-width", 2.5)
        .style("fill-opacity", 0.3)
      let text = "";
      text = `<p>${d.idxs.length} samples in this cluster</p>
      <p>The average prediction score is ${Number(d.prediction_score).toFixed(4)}</p>`
      self.highlightGrid(d.idxs)
      // self.showTooltip({ top, left, content: text })
    }, 500)
    /*
    setTimeout(() => {
      if (hover_grid_item == d) {
        show_detail();
      }
    }, 500)*/
  };
  const on_mouseout = function (d) {
    clearTimeout(timer)
    d3
      .select(this).style("stroke-width", 1)
      .style("fill-opacity", 0.2)
    if (!d.idxs) return
    self.highlightGrid(null)
    // self.hideTooltip()
  };

  const on_click = function (d) {
    if (!d.idxs) return
    if (!d.is_current_sample) {
      bus.$emit("brush_grid", d.idxs)
      d.is_current_sample = 1
    } else {
      bus.$emit("brush_grid", null)
      d.is_current_sample = 0
    }
  };

  svg
    .append("rect")
    .attr("width", (d) => d.width + 0.5)
    .attr("height", (d) => d.height + 0.5)
    //.attr("rx", 3)
    //.attr("ry", 3)
    .style("fill", (d) => self.getWordFillColor(d))
    .style("stroke", (d) => self.getWordFillColor(d))
    .style("fill-opacity", (d) => 0.2)
    .style("stroke-width", 1) //2)
    .style("stroke-opacity", 0.8) //2)
    .on("mouseover", on_mouseover)
    .on("mouseout", on_mouseout)
    .on("click", on_click);

  svg.append("g").attr("class", "wordcloud");
  //.attr('transform', d => `translate(5, 5)`)

  let cluster_edges = self.edges.filter(
    (d) => d.y1 && d.y2 && d.y3 && d.y4 && d.y4 != d.y3 && d.y1 != d.y2
  )
/*
  let thin_edge = DAG.selectAll(".thin-cluster-edge")
    .data(cluster_edges.filter((d) => d.y2 - d.y1 < 100 && d.y1 < d.y3))
    .enter()
    .append("path")
    .attr("class", "thin-cluster-edge")
    .attr("d", (d) => {
      let mid = (d.x1 + d.x2) / 2;
      return `M${d.x1 + 1} ${(d.y1 + d.y2) / 2} C${mid} ${
        (d.y1 + d.y2) / 2
      } ${mid} ${(d.y3 + d.y4) / 2} ${d.x2 - 1} ${(d.y3 + d.y4) / 2}`;
    })
    .style("stroke-width", (d) => d.y2 - d.y1)
    .style("stroke", d3.interpolateRgb("white", "rgb(172,196,220)")(0.35))
    .style("opacity", 1) //d => ((d.y2-d.y1) / max_width) ** 0.8 * 0.8)
    .style("fill", "none");
*/
  let thin_edge_2 = DAG.selectAll(".cluster-edge")
    .data(cluster_edges.filter((d) => 1))//d.y2 - d.y1 < 100 && d.y1 >= d.y3))
    .enter()
    .append("path")
    .attr("class", "cluster-edge")
    .attr("d", (d) => {
      let mid = (d.x1 + d.x2) / 2;
      return `M${d.x1} ${(d.y1 + d.y2) / 2} C${mid} ${
        (d.y1 + d.y2) / 2
      } ${mid} ${(d.y3 + d.y4) / 2} ${d.x2} ${(d.y3 + d.y4) / 2}`;
    })
    .style("stroke-width", (d) => d.y2 - d.y1)
    .style("stroke", "rgb(172,196,220)")
    .style("opacity", 0.35) //d => ((d.y2-d.y1) / max_width) ** 0.8 * 0.8)
    .style("fill", "none");
/*
  let thick_edge = DAG.selectAll(".thick-cluster-edge")
    .data(cluster_edges.filter((d) => d.y2 - d.y1 >= 100))
    .enter()
    .append("path")
    .attr("class", "thick-cluster-edge")
    .attr("d", (d) => {
      let mid = (d.x1 + d.x2) / 2;
      return `M${d.x1} ${d.y1}L${d.x1} ${d.y2} C${mid} ${d.y2} ${mid} ${d.y4} ${d.x2} ${d.y4} L${d.x2} ${d.y3} C${mid} ${d.y3} ${mid} ${d.y1} ${d.x1} ${d.y1} z`;
    })
    .style("stroke-width", 0)
    .style("fill", "rgb(172,196,220)")
    .style("opacity", 0.35);*/

  //console.log('current_layers', self.current_layers)
  let layers = DAG.selectAll(".layer")
    .data(self.current_layers)
    .enter()
    .append("g")
    .attr("class", "layer")
    .attr("transform", (d) => `translate(${d.x + d.width / 2 - 80},${30})`);

  layers
    .append("text")
    .attr("fill", "#111")
    .text((d) => {
      if (d.layer_id == 1) {
        return "Input";
      } else if (d.layer_id > self.layers.length) {
        return "Prediction";
      } else {
        return `Layer ${d.layer_id - 1}`;
      }
    })
    .style("font-size", "18px")
    .attr("text-anchor", "middle");
}


export function layoutSentenceDAGNodes(state) {
  let width = state.DAG.width + 60
  let height = state.DAG.height - 100
  let unique_layers = [...new Set(state.neuron_clusters.map(d => d.layer_id))]
  unique_layers.sort((a, b) => a - b)
  let layerlen = unique_layers.length
  let layers = unique_layers.map(d => ({
    layer_id: d,
    scale: (d.length ? 0.5 : 1.0) / layerlen,
  }))
  let edge_width = width / layerlen * (1 - state.config.network.node_scale)
  let padding = state.config.network.node_padding
  state.neuron_clusters.forEach(d => { d.width = 0; d.height = 0; })
  for (let i = 0; i < layers.length; ++i) {
    layers[i].width = width * layers[i].scale
    layers[i].x = i == 0 ? 50 : (layers[i - 1].x + layers[i - 1].width)
    layers[i].clusters = state.neuron_clusters.filter(d => d.layer_id == layers[i].layer_id)
    let totsize = layers[i].clusters.map(d => d.size).reduce((a, b) => a + b, 0)
    let h0 = 50
    let n = layers[i].clusters.length
    let lh = height - (n - 1) * padding
    layers[i].clusters.forEach(d => {
      d.scale = 1.0 / layers[i].clusters.length
      d.height = Math.min(Math.max(padding * 3.5, lh * d.size / totsize), lh / 2)
      d.show_cloud = true
      d.width = layers[i].width - edge_width
      d.x = layers[i].x
    })
    for (let j = 0; j < layers[i].clusters.length; ++j) {
      layers[i].clusters[j].y = j == 0 ? 50 : (layers[i].clusters[j - 1].y + layers[i].clusters[j - 1].height + padding)
    }
    if (layers[i].clusters[n - 1].y + layers[i].clusters[n - 1].height > height) {
      let totsize = layers[i].clusters.map(d => d.height).reduce((a, b) => a + b, 0)
      layers[i].clusters.forEach(d => d.height *= lh / totsize)
      for (let j = 0; j < layers[i].clusters.length; ++j) {
        layers[i].clusters[j].y = j == 0 ? h0 : (layers[i].clusters[j - 1].y + layers[i].clusters[j - 1].height + padding)
      }
    } else if (layers[i].clusters[n - 1].y + layers[i].clusters[n - 1].height < height) {
      let delta = (height - layers[i].clusters[n - 1].y - layers[i].clusters[n - 1].height) / 2
      for (let j = 0; j < layers[i].clusters.length; ++j) {
        layers[i].clusters[j].y += delta
      }
    }
  }
  state.current_layers = layers
}

export async function paintWordcloud(self) {
  let DAG = d3.select("#network-svg")
  let all_words = []
  for (let k = 0; k < self.neuron_clusters.length; ++k) {
    let d = self.neuron_clusters[k]
    if (!d.show_cloud) continue

    const text_padding = 0
    const max_size = d3
      .scalePow()
      .exponent(1.5)
      .domain([40, 150])
      .range([21, 27])
    let font_size_range = [16, Math.min(28, max_size(d.height))]
    let font_value_range = [
      Math.min(...d.retained_keywords.map((e) => e[1])),
      Math.max(...d.retained_keywords.map((e) => e[1])),
    ]
    if (font_value_range[0] == font_value_range[1]) {
      font_size_range[0] = font_size_range[1]
    }

    let keyword_size = stepSizeScale(
      font_size_range,
      d.retained_keywords.map((e) => e[1]),
      4,
      [1, 3, 5]
    );

    let keywords = d.retained_keywords.map((e) => ({
      size: keyword_size(e[1]),
      text: e[0],
      type: 1,
      opacity: 0.9,
      weight: e[2] == "self" ? "bold" : "bold",
      fontstyle: e[2] == "self" ? "normal" : "italic",
      center: e[2] == "self",
    }))

    keywords = keywords.sort((a, b) => {
      if (a.center != b.center) {
        return b.center - a.center;
      } else {
        return b.size - a.size;
      }
    })

    if (d.layer_id == 12) {
      keywords = keywords.filter((d) => d.text != "enjoyment");
    }
    const font_color = "#6D6761"

    if (d.layer_id == 1) {
      let words = keywords[0].text.split("$");
      let font_size = 20
      all_words.push({
        id: d.id,
        "x": d.x + d.width / 2,
        "y": d.y + d.height / 2 + text_padding,
        "font-size": `${font_size + 4}px`,
        "font-weight": "bold",
        "font-style": "normal",
        "fill": font_color,
        "opacity": 0.9,
        "text-anchor": "middle",
        "font-family": "Roboto: sans-serif",
        text: words[0],
      })
      all_words.push({
        id: d.id,
        "x": d.x + d.width / 2,
        "y": d.y + d.height / 2 + font_size + text_padding,
        "font-size": `${font_size - 2}px`,
        "font-weight": "bold",
        "font-style": "normal",
        "fill": font_color,
        "text-anchor": "middle",
        "font-family": "Roboto: sans-serif",
        "opacity": 0.9,
        text: words[1],
      })
    } else {
      await wordcloud()
        .size([d.width, d.height - text_padding])
        .data(keywords)
        .fontSize((d) => d.size)
        .font("Roboto")
        .padding(6)
        .flexible(1)
        .fontWeight((d) => d.weight)
        .start((words) => {
          words = words.filter((d) => d.display)
          all_words = all_words.concat(
            words.map(e => ({
              id: d.id,
              "x": d.x + e.x,
              "y": d.y + e.y + text_padding,
              "font-size": `${e.size}px`,
              "font-weight": e.weight,
              "font-style": e.style,
              "fill": font_color,
              "font-family": "Roboto: sans-serif",
              "text-anchor": "start",
              "opacity": e.opacity,
              text: e.text,
            }))
          )
        })
    }
  }


  let timer = null
  const on_mouseover = function (d) {
    clearTimeout(timer)
    timer = setTimeout(() => {
      let current_ids = all_words
        .filter(e => e.text == d.text)
        .map(e => e.id)
      current_ids = new Set(current_ids)

      DAG.selectAll(".word")
        .filter(e => e.text == d.text)
        .style("fill", "#424242")
        .style("opacity", 1)

      DAG.selectAll(".cluster-edge")
        .filter(e => current_ids.has(e.source) && current_ids.has(e.target))
        .style("opacity", .6)
    }, 250)
  };
  const on_mouseout = function (d) {
    clearTimeout(timer)
    DAG.selectAll(".word")
      .style("fill", d => d["fill"])
      .style("opacity", d => d["opacity"])
    
    let current_ids = all_words
      .filter(e => e.text == d.text)
      .map(e => e.id)
    current_ids = new Set(current_ids)
    DAG.selectAll(".cluster-edge")
      .filter(e => current_ids.has(e.source) && current_ids.has(e.target))
      .style("opacity", .35)
  };

  DAG.selectAll(".word")
    .data(all_words).enter()
    .append("text")
    .attr("class", "word")
    .attr("x", d => d.x)
    .attr("y", d => d.y)
    .attr("font-size", d => d["font-size"])
    .attr("font-weight", d => d["font-weight"])
    .attr("font-style", d => d["font-style"])
    .style("fill", d => d["fill"])
    .style("font-family",  d => d["font-family"])
    .style("text-anchor", d => d["text-anchor"])
    .style("opacity", d => d["opacity"])
    .text(d => d.text)
    .on("mouseover", on_mouseover)
    .on("mouseout", on_mouseout)
    .on("click", async function(d){
      await self.fetchWordIndex(d.text)
      let idxs = self.getWordIdxs(d.text)
      if (idxs.length >= 8) {
        bus.$emit('add_word_DAG', d.text)
      }
    })
}