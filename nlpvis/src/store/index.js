import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'
import { layoutDAGEdges, layoutSentenceDAGNodes } from '../layout/word_dag_layout'

Vue.use(Vuex)

let cache = {
  corpus_layers: null,
}

export default new Vuex.Store({
  state: {
    server_url: 'http://nlpvis-demo.thuvis.org',//'./'
    DAG: {},
    linechart: {},
    scatterplot: {},
    bubble: {
      left: 40,
      top: 10,
    },
    config: {
      linechart: { scale: 0.3, padding: 20 },
      DAG: { padding: { top: 5, left: 24 } },
      network: {
        padding: { left: 20, top: 20, right: 20, bottom: 20 },
        node_scale: 0.6,
        node_padding: 20,
      },
      data_table: {
        max_table_len: 3000,
        min_text_len: 30,
        target_text_len: 45,
      }
    },
    color_scheme: ["#d7191c", "#1a9641", "#ff7f0e", "#1f77b4" ],//["#d7191c", "#1a9641", "#9467bd", "#ff7f0e"],
    neuron_clusters: [],
    edges: [],
    showed_layers: [0, 3, 6, 9, 10, 12],
    layers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    current_layers: [],
    current_viewtype: 'sentence',
    layout: null,
    sentenceclusters: [],
    all_samples: [],
    hover_word: null,
    hover_timer: null,
    current_samples: [],
    word_indexs: {},
    word_index_pos: {},
    filtered_words: [],
    labels: [],
    filter_info: {},
    tooltip: {
      top: 0,
      left: 0,
      show: false,
      content: '',
    },
    selected_class: [1, 0],
  },
  getters: {
    positiveColor: (state) => state.color_scheme[state.selected_class[0]],
    negativeColor: (state) => state.color_scheme[state.selected_class[1]],
    neutralColor: (state) => 'gray',
    neutralColorGray: (state) => "#777",
    fontSizeRange: (state) => [15, 35],
    getSample: (state) => (idx) => {
      return state.all_samples[idx]
    },
    checkWordIndex: (state) => (key) => {
      return !!state.word_indexs[key]
    },
    getLabelColor: (state) => (label) => {
      for (let i = 0; i < state.labels.length; ++i) {
        if (state.labels[i].label == label) {
          return state.color_scheme[i]
        }
      }
      return 'gray'
    },
    selected_label: (state) => {
      return state.selected_class.map(d => state.labels[d].label)
    },
    getSamples: (state) => (idxs) => {
      let ret
      const max_len = state.config.data_table.max_table_len
      const target_len = state.config.data_table.target_text_len
      //console.log(idxs)
      if (idxs && idxs.length > 0) {
        ret = idxs.map(idx => state.all_samples[idx])
          .sort((a, b) => Math.abs(a.text.length - target_len) - Math.abs(b.text.length - target_len))
          .slice(0, max_len)
      } else {
        ret = state.all_samples
          .filter(d => d.text.length > state.config.data_table.min_text_len)
          .slice(0, max_len)
          .sort((a, b) => Math.abs(a.text.length - target_len) - Math.abs(b.text.length - target_len))
      }
      //console.log(ret)
      return ret
    },
    getWordIdxs: (state) => (key) => {
      return state.word_indexs[key]
    },
    view_samples: (state) => {
      let oldidxs = new Set(state.current_samples.map(d => d.index))
      let newidxs = new Set()
      for (let word of state.filtered_words) {
        for (let idx of state.word_indexs[word]) {
          newidxs.add(idx)
          if (oldidxs.has(idx)) {
            oldidxs.delete(idx)
          }
        }
      }
      oldidxs = [...oldidxs]
      oldidxs.sort((a, b) => a - b)
      let idxs = [...newidxs].concat(oldidxs)
      function highlight(texts) {
        let words = []
        texts = texts.split(' ')
        for (let text of texts) {
          let flag = 1
          for (let word of state.filtered_words) {
            if (text.indexOf(word) != -1) {
              flag = 0
              let p = text.indexOf(word)
              if (p > 0) {
                words.push([text.slice(0, p), false])
              }
              words.push([word, true])
              if (word.length + p < text.length) {
                words.push([text.slice(word.length + p), false])
              }
              break
            }
          }
          if (flag) {
            words.push([text, false])
          }
        }
        return words
      }
      let ret = idxs.map(idx => state.all_samples[idx])
        .map(d => {
          let words = highlight(d.text)
					let x1 = d.score[state.selected_class[0]]
					let x2 = d.score[state.selected_class[1]]
          return {
            index: d.index,
            text: words,
            pred: d.pred,
            label: d.label,
            score: x1 / (x1 + x2),
            show_filter: false,
          }
        })
      if (newidxs.size > 0) {
        ret[0].show_filter = true
        ret = ret.slice(0, newidxs.size)
          .concat(state.filter_info.sentences
            .filter(d => !newidxs.has(d.index) && d.text.indexOf(` ${state.filtered_words[0]} `) != -1 && d.index >= 5000)
            .map(d => ({
              index: d.index,
              text: highlight(d.text),
              pred: 'N/A',
              label: d.label,
              score: 'N/A',
              show_filter: false,
            }))
            .sort((a, b) => {
              if (a.index == 60312) return -1
              else if (b.index == 60312) return 1
              return (a.index - b.index)
            })
          )
          .concat(ret.slice(newidxs.size))
      }
      return ret.slice(0, 500)
    }
  },
  mutations: {
    addHoverTimer(state, timer) {
      clearTimeout(state.hover_timer)
      state.hover_timer = timer
    },
    removeHoverTimer(state) {
      clearTimeout(state.hover_timer)
    },
    setHoverWord(state, key) {
      state.hover_word = key
    },
    updatePosition(state, payload) {
      const padding = state.config.network.padding
      const width = payload.network.width - padding.left - padding.right
      const height = payload.network.height - padding.top - padding.bottom
      state.bubble.width = payload.bubble.width
      state.bubble.height = payload.bubble.height
      state.linechart.padding = state.config.linechart.padding
      state.linechart.width = payload.linechart.width - state.linechart.padding
      state.linechart.height = payload.linechart.height - state.linechart.padding
      state.linechart.x = state.linechart.padding
      state.linechart.y = state.linechart.padding
      state.DAG.width = width
      state.DAG.height = height
      state.DAG.x = state.config.DAG.padding.left
      state.DAG.y = state.config.DAG.padding.top
    },
    setViewtype(state, type) {
      state.current_viewtype = type
    },
    setScatterplot(state, payload) {
      state.scatterplot = payload.data
    },
    setLayers(state, data) {
      state.layers = data
    },
    calcEdges(state, layers) {
      layoutDAGEdges(state, layers)
    },
    calcDAG(state) {
      if (state.current_viewtype == 'word') {
        layoutSentenceDAGNodes(state)
        this.commit('calcEdges', state.current_layers)
      }
    },
    changeFilteredWord(state, { key, sentences, pos, neg }) {
      const idx = state.filtered_words.indexOf(key)
      if (idx != -1) {
        state.filtered_words.splice(idx, 1)
      } else {
        state.filtered_words = [key]
        state.filter_info = {
          sentences, pos, neg, word: key,
        }
      }
    },
    setNetwork(state, payload) {
      if (state.current_viewtype == 'sentence') {
        state.layout = payload.data.layout
      } else if (state.current_viewtype == 'word') {
        state.neuron_clusters = payload.data.neuron_clusters
        state.edges = payload.data.edges
      }
      //console.log(state.neuron_clusters)
    },
    setAllSamples(state, data) {
      state.all_samples = data.sort((a, b) => a.id - b.id)
    },
    setCurrentSamples(state, data) {
      state.current_samples = data
    },
    setWordIndex(state, { key, idxs, pos }) {
      state.word_indexs[key] = idxs
      state.word_index_pos[key] = pos
    },
    showTooltip(state, { top, left, content }) {
      state.tooltip.top = top + 10
      state.tooltip.left = left + 10
      state.tooltip.content = content
      state.tooltip.show = true
    },
    setConfusionMatrix(state, { matrix, labels, label_names, selection }) {
      state.selected_class = selection
      state.labels = labels.map((d, i) => ({ label: d, name: label_names[i] }))
      state.matrix = matrix
    },
    hideTooltip(state) {
      state.tooltip.show = false
    }
  },
  actions: {
    async changeCurrentSamples({ commit, getters }, idxs) {
      commit('setCurrentSamples', getters.getSamples(idxs))
    },
    async changeFilteredWord({ commit, state }, key) {
      const resp = await axios.post(`${state.server_url}/api/word_sentences`, { word: key })
      const { sentences, pos, neg } = resp.data
      commit('changeFilteredWord', { key, sentences, pos, neg })
    },
    async fetchWordIndex({ commit, state }, key) {
      if (!!state.word_indexs[key]) {
        return
      }
      const resp = await axios.post(`${state.server_url}/api/word`, { word: key })
      const idxs = resp.data.idxs
      const pos = resp.data.pos
      commit('setWordIndex', { key, idxs, pos })
    },
    async fetchLayerInfo({ commit, state }, { idxs, attrs }) {
      let resp
      if (idxs == null && cache.corpus_layers != null) {
        resp = cache.corpus_layers
      } else {
        resp = await axios.post(`${state.server_url}/api/layers`, { idxs, attrs })
        if (idxs == null) {
          cache.corpus_layers = resp
        }
      }
      commit('setLayers', resp.data)
    },
    async fetchDAG({ commit, state }, req) {
      let resp = await axios.post(`${state.server_url}/api/networks`, req)
      commit('setNetwork', resp)
      commit('calcDAG')
    },
    async fetchAllSample({ commit, state, getters }) {
      let resp = await axios.post(`${state.server_url}/api/all_sentences`, {})
      // console.log(resp.data.sentences)
      commit('setAllSamples', resp.data.sentences)
      resp = await axios.post(`${state.server_url}/api/confusion_matrix`, {})
      commit('setConfusionMatrix', resp.data)
      commit('setCurrentSamples', getters.getSamples(null))
    }
  },
  modules: {
  }
})
