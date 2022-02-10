<template>
  <v-app style="position: relative" ref="mainview">
    <info-tooltip
      :left="tooltip.left"
      :top="tooltip.top"
      :show="tooltip.show"
      :content="tooltip.content"
    >
    </info-tooltip>
    <!--v-app-bar dense flat app color="grey darken-4" dark>
      <div class="d-flex align-center headline">DeepNLP Vis</div>
      <v-spacer></v-spacer>
      <div class="d-flex align-center">SST2, 67349 training samples</div>
      <v-btn icon>
        <v-icon>mdi-chevron-down</v-icon>
      </v-btn>
    </v-app-bar-->

    <v-content>
      <v-container fluid class="core-view fill-height">
        <v-row class="fill-height">
          <v-col cols="4" class="fill-height py-0 pr-0">
            <v-card class="nlpvis-view" height="40%" v-intro-step="2" v-intro="'In the distribution view, user can identify important samples and words for further analysis.'">
              <!--v-row class="pa-1 pl-4 nlpvis-title"> Corpus </v-row-->
              <v-toolbar dense flat>
                <v-btn-toggle dense class="nlpvis-tab"
                  v-model="left_top_view"
                >
                  <v-btn value="matrix">
                    <span class="nlpvis-title">Class</span>
                  </v-btn>

                  <v-btn value="corpus">
                    <span class="nlpvis-title">Distribution</span>
                  </v-btn>
                </v-btn-toggle>
                <v-spacer></v-spacer>
                <v-checkbox
                  v-model="is_lasso_select_all"
                  style="padding-top: 12px"
                  color="darkgray"
                >
                <template v-slot:prepend>
                  <v-img width="115px" height="24px" src="./assets/figure1.png"></v-img>
                </template>
                </v-checkbox>
                <!--v-radio-group
                  v-model="row"
                  row
                >
                  <template v-slot:label>
                    <div>
                      <svg version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                        viewBox="0 0 512 512" style="enable-background:new 0 0 512 512; width: 16px; height: 16px; opacity: .5" xml:space="preserve">
                      <g>
                        <g>
                          <g>
                            <path d="M453.072,140.002l24.448-20.656c-7.542-8.861-15.869-17.022-24.88-24.384l-20.304,24.704
                              C439.842,125.811,446.782,132.617,453.072,140.002z"/>
                            <path d="M424.528,75.202c-9.74-5.814-19.835-11.014-30.224-15.568l-12.8,29.312c9.156,3.984,18.045,8.559,26.608,13.696
                              L424.528,75.202z"/>
                            <path d="M512,208.002c-0.068-20.623-5.094-40.927-14.656-59.2l-28.464,14.624c7.238,13.748,11.052,29.039,11.12,44.576
                              c0,79.408-100.48,144-224,144c-10.944,0-21.648-0.656-32.16-1.6c-0.758-35.232-29.934-63.179-65.166-62.421
                              c-20.341,0.438-39.253,10.549-50.914,27.221C60.176,287.906,32,248.722,32,208.002c0-79.408,100.48-144,224-144
                              c32.916-0.152,65.664,4.703,97.12,14.4l9.6-30.528C328.16,37.19,292.173,31.838,256,32.002c-141.152,0-256,78.96-256,176
                              c0,53.92,35.904,104.768,96.624,137.888c-0.313,2.025-0.522,4.065-0.624,6.112c-0.057,35.284,28.501,63.934,63.786,63.99
                              c21.914,0.035,42.317-11.164,54.054-29.67c34.527,11.019,58.016,43.037,58.16,79.28v14.4h32v-14.4
                              c-0.055-30.856-12.454-60.406-34.432-82.064C404.4,378.626,512,301.906,512,208.002z M160,384.002c-17.673,0-32-14.327-32-32
                              c0-17.673,14.327-32,32-32s32,14.327,32,32C192,369.675,177.673,384.002,160,384.002z"/>
                          </g>
                        </g>
                      </g>
                      </svg>
                    </div>
                  </template>
                  <v-radio value="radio-1">
                    <template v-slot:label>
                      <div>Allow
                      </div>
                    </template>
                  </v-radio>
                  <v-radio value="radio-2">
                    <template v-slot:label>
                      <div>Of course it's <strong class="success--text">Google</strong></div>
                    </template>
                  </v-radio>
                </v-radio-group-->

                <!--v-icon size="18"
                  class="px-2"
                  :color="show_trending_down ? 'orange darken-2' : '#777'"
                  @click="show_trending_down = !show_trending_down"
                >
                  mdi-trending-down
                </v-icon-->
              </v-toolbar> 
              <hr/>
              <svg id="overview-svg" class="ma-1" v-intro-step="1" v-intro="'Click on an element in the matrix to start the analysis of any two classes.'">
                <g id="overview-g" v-show="left_top_view == 'corpus'"></g>
                <word-group v-show="left_top_view == 'corpus'"
                  :tooltip="enable_tooltip"
                  :transform="`translate(${bubble.left},${0})`"
                  :data="overviewWords"
                  :interaction="enable_word_filter ? 'filter' : 'DAG'"
                ></word-group>
                <g id="matrix-g" v-show="left_top_view == 'matrix'">
                  <g transform="translate(30, 10)">
                    <image x="0" y="18" width="320" height="10" preserveAspectRatio="none" xlink:href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAABCAYAAAAxWXB3AAAAH0lEQVQ4T2P8////fwYGBgYoNUpDgmM0HEbDYUSkAwCp0H+QW+DsjwAAAABJRU5ErkJggg=="></image>
                    <g transform="translate(0,28)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><g class="tick" opacity="1" transform="translate(0.5,0)"><line stroke="currentColor" y2="6" y1="-10"></line><text fill="currentColor" y="9" font-size="18px" font-family="Arial" dy="0.71em">0.0</text></g><g class="tick" opacity="1" transform="translate(143.5,0)"><line stroke="currentColor" y2="6" y1="-10"></line><text fill="currentColor" y="9" font-size="18px" font-family="Arial" dy="0.71em">0.2</text></g><g class="tick" opacity="1" transform="translate(203.5,0)"><line stroke="currentColor" y2="6" y1="-10"></line><text font-size="18px" font-family="Arial" fill="currentColor" y="9" dy="0.71em">0.4</text></g><g class="tick" opacity="1" transform="translate(249.5,0)"><line stroke="currentColor" y2="6" y1="-10"></line><text fill="currentColor" y="9" font-size="18px" font-family="Arial" dy="0.71em">0.6</text></g><g class="tick" opacity="1" transform="translate(287.5,0)"><line stroke="currentColor" y2="6" y1="-10"></line><text fill="currentColor" font-size="18px" font-family="Arial" y="9" dy="0.71em">0.8</text></g><text x="0" y="-16" font-size="18px" font-family="Arial" fill="currentColor" text-anchor="start" font-weight="bold">Percentage</text></g>
                  </g>
                </g>
              </svg>
            </v-card>
            <v-card class="nlpvis-view pa-1 mt-2" min-height="calc(53% - 48px)"
            v-intro="'In the sample list, user can examine multiple samples in terms of their text content, class labels, and prediction scores.'"
            v-intro-step="4">
              <v-toolbar dense flat>
                <v-btn-toggle dense class="nlpvis-tab"
                  v-model="left_bottom_view"
                >
                  <v-btn value="samples">
                    <span class="nlpvis-title">Sample List</span>
                  </v-btn>
                </v-btn-toggle>
                <v-spacer></v-spacer>
                <template v-if="!search_expand">
                  <span class="pa-1 nlpvis-title disabled"></span>
                  <span :key="`legend${index}`" v-for="(t, index) in labels" class="color-legend" :style="`--color: ${color_scheme[index]}`">
                    {{ t.name }}
                  </span>
                </template>
                <template v-else>
                <v-combobox
                  v-model="chips"
                  chips
                  clearable
                  label="Enter a word or sample number here"
                  multiple
                  solo
                >
                  <template v-slot:selection="{ attrs, item, selected }">
                    <v-chip
                      v-bind="attrs"
                      :input-value="selected"
                      close
                      @click="clickSearchItem(item)"
                      @contextmenu.prevent="rightClickSearchItem(item)"
                      @click:close="removeSearchItem(item)"
                    >
                      <strong>{{ item }}</strong>&nbsp;
                      <span>{{ isNaN(+item) ? '(Word)' : '(Sample)'}}</span>
                    </v-chip>
                  </template>
                </v-combobox>
                </template>
                <v-btn icon @click="onClickSearch()">
                  <v-icon size="26">mdi-magnify</v-icon>
                </v-btn>
              </v-toolbar> 
              <hr/>
              <v-simple-table
                dense
                ref="simpletable"
                style="
                  overflow-x: hidden;
                  max-height: calc(52vh - 96px);
                  max-width: 100%;
                  overflow-y: scroll;
                "

              >
                <template v-slot:default>
                  <thead>
                    <tr>
                      <th class="text-left nlpvis-text pa-1" style="white-space: nowrap">
                        ID<v-icon size="18" style="color: #444"
                          >mdi-menu-swap</v-icon
                        >
                      </th>
                      <th class="text-left nlpvis-text pa-1">
                        Passage<v-icon size="18" style="color: #444"
                          >mdi-menu-swap</v-icon
                        >
                      </th>
                      <th class="text-left nlpvis-text pa-1" style="white-space: nowrap">
                        Label<v-icon size="18" style="color: #444"
                          >mdi-menu-swap</v-icon
                        >
                      </th>
                      <th class="text-left nlpvis-text pa-1" style="white-space: nowrap">
                        Score<v-icon size="18" style="color: #444"
                          >mdi-menu-swap</v-icon
                        >
                      </th>
                      <th class="text-left nlpvis-text pa-1" style="white-space: nowrap">
                        Correctness<v-icon size="18" style="color: #444"
                          >mdi-menu-swap</v-icon
                        >
                      </th>
                    </tr>
                  </thead>
                  <tbody style="
                  overflow-x: hidden;
                  max-height: calc(52% - 96px);
                  overflow-y: scroll;
                "
>
                    <template v-for="item in view_samples">
                      <tr :key="`filter${item.index}`" v-if="item.show_filter">
                        <td class="pa-1 nlpvis-text"></td>
                        <td class="pa-1 nlpvis-text" style="color: #777">
                          There are {{filter_info.pos}} negative samples and {{filter_info.neg}} positive samples include "{{filter_info.word}}"  
                        </td>
                        <td class="pa-1 nlpvis-text" style="white-space: nowrap">
                        </td>
                        <td class="pa-1 nlpvis-text">
                        </td>
                        <td class="pa-1 nlpvis-text">
                        </td>
                      </tr>
                      <tr
                        :key="item.index"
                        @mouseenter="highlightGrid([item.index])"
                        @mouseleave="highlightGrid(null)"
                        @click="addSentenceDAG(item.index)"
                      >
                        <td class="pa-1 nlpvis-text">{{ item.index }}</td>
                        <td class="pa-1 nlpvis-text" :style="{ color: item.pred == 'N/A' ? '#777' : 'black' }">
                          <span
                            v-for="(word, index) in item.text"
                            :key="index"
                            :style="{
                              background: word[1] ? 'orange' : 'none',
                            }"
                          >
                            {{ word[0] }}
                          </span>
                        </td>
                        <td class="pa-1 nlpvis-text" style="white-space: nowrap">
                          {{ item.label }}
                          <span
                            style="
                              display: inline-block;
                              width: 16px;
                              height: 16px;
                            "
                            :style="{
                              'background-color': getLabelColor(item.label),
                            }"
                          ></span>
                        </td>
                        <td class="pa-1 nlpvis-text">
                          {{ isNaN(item.score) ? 'N/A' : Number(item.score).toFixed(2) }}
                        </td>
                        <td class="pa-1 nlpvis-text">
                          <v-icon v-if="item.pred == item.label"
                            >mdi-check</v-icon size='16'
                          >
                          <v-icon v-else-if="item.pred != 'N/A'">mdi-close</v-icon>
                        </td>
                      </tr>
                    </template>
                  </tbody>
                </template>
              </v-simple-table>
            </v-card>
            <v-card class="nlpvis-script pa-1 mt-2" height="calc(7% + 8px)">
              <v-toolbar flat dense style="margin-top: 2.5%">
              <v-select
                :items="datasets"
                label="SST-2"
                disabled
                class="nlpvis-text"
                outlined
              ></v-select>
              <v-select
                :items="models"
                label="12-layer BERT"
                disabled
                class="nlpvis-text ml-2"
                outlined
              ></v-select>
              </v-toolbar>
            </v-card>
          </v-col>
          <v-col cols="8" class="fill-height py-0">
            <v-card class="nlpvis-view" height="40%" v-intro-step="3" v-intro="'In the word contribution view, it shows how words are leveraged at different layers for prediction.'">
              <v-toolbar dense flat>
                <v-btn-toggle dense class="nlpvis-tab"
                  v-model="right_top_view"
                >
                  <v-btn value="word">
                    <span class="nlpvis-title">Word Contribution</span>
                  </v-btn>
                </v-btn-toggle>
                <v-spacer></v-spacer>
                <v-icon size="26"
                  class="px-2"
                  :color="show_trending_down ? '#333' : '#ccc'"
                  @click="show_trending_down = !show_trending_down"
                >
                  mdi-trending-down
                </v-icon>
                <v-icon size="26"
                  class="px-2"
                  :color="show_trending_up ? '#333' : '#ccc'"
                  @click="show_trending_up = !show_trending_up"
                >
                  mdi-trending-up
                </v-icon>
                <!--v-icon size="18"
                  class="px-2"
                  :color="enable_tooltip ? 'orange darken-2' : '#777'"
                  @click="enable_tooltip = !enable_tooltip"
                >
                  mdi-message
                </v-icon>
                <v-icon size="18"
                  @click="enable_word_filter = !enable_word_filter"
                  class="px-2"
                  :color="enable_word_filter ? 'orange darken-2' : '#777'"
                >
                  mdi-filter
                </v-icon-->
                <!--v-menu
                  v-model="enable_word_attr"
                  :close-on-content-click="false"
                  :nudge-width="200"
                  offset-x
                >
                  <template v-slot:activator="{ on, attrs }">
                    <v-btn
                      x-small
                      outlined
                      v-bind="attrs"
                      class="mx-1"
                      color="orange darken-2"
                      v-on="on"
                    >
                      word weight
                      <v-icon>mdi-chevron-down</v-icon>
                    </v-btn>
                  </template>

                  <v-card>
                    <v-subheader>Word ranking weight</v-subheader>
                    <v-subheader>Uncertainty</v-subheader>
                    <v-card-text class="py-0">
                      <v-slider
                        v-model="uncertainty_weight"
                        min="0"
                        max="10"
                      ></v-slider>
                    </v-card-text>
                    <v-subheader>Frequency</v-subheader>
                    <v-card-text class="py-0">
                      <v-slider
                        v-model="frequency_weight"
                        min="0"
                        max="10"
                      ></v-slider>
                    </v-card-text>

                    <v-subheader>Entropy</v-subheader>
                    <v-card-text class="py-0">
                      <v-slider
                        v-model="entropy_weight"
                        min="0"
                        max="10"
                      ></v-slider>
                    </v-card-text>

                    <v-card-text>
                      <v-btn text @click="enable_word_attr = false"
                        >Cancel</v-btn
                      >
                      <v-btn color="primary" text @click="saveWordAttr()"
                        >Save</v-btn
                      >
                    </v-card-text>
                  </v-card>
                </v-menu-->
              </v-toolbar>
              <hr/>
              <svg id="linechart-svg" class="ma-1" >
                <g id="linechart-g" v-show="is_wordview_init">
                  <template v-if="show_trending_up">
                    <path transform="translate(0, 0)"
                          v-for="(path, index) in trending_up_lines"
                          :key="index" :d="path" fill="none"
                          stroke="#dbdedb" stroke-width="15px"
                          style="opacity: 1"
                    >
                    </path>
                    <g transform="translate(0, 0)" 
                          :key="index" 
                          v-for="(underlines, index) in trending_up_underlines">
                      <line v-for="(line, index2) in underlines"
                            :key="index2" fill="none"
                            style="opacity: 1; stroke-width: 3px; stroke:gray;"
                            :x1="line.x1"
                            :x2="line.x2"
                            :y1="line.y1"
                            :y2="line.y2"
                      >
                      </line>
                    </g>
                  </template>
                  <template v-if="show_trending_down">
                    <path transform="translate(0, 0)"
                          v-for="(path, index) in trending_down_lines"
                          :key="index" :d="path" fill="none"
                          stroke="#dbdedb" stroke-width="15px"
                          style="opacity: 1"
                    >
                    </path>
                    <g transform="translate(0, 0)" 
                          :key="index" 
                          v-for="(underlines, index) in trending_down_underlines">
                      <line v-for="(line, index2) in underlines"
                            :key="index2" fill="none"
                            style="opacity: 1; stroke-width: 3px; stroke:gray;"
                            :x1="line.x1"
                            :x2="line.x2"
                            :y1="line.y1"
                            :y2="line.y2"
                      >
                      </line>
                    </g>
                  </template>
                  <word-group
                    :tooltip="enable_tooltip"
                    :transform="`translate(${linechart.padding},${0})`"
                    :data="infolossWords"
                    :highlight="
                      enable_word_highlight && current_viewtype == 'sentence'
                    "
                    :highlight_text="current_text"
                    :interaction="enable_word_filter ? 'filter' : 'DAG'"
                  ></word-group>
                </g>
              </svg>
            </v-card>
            <v-card class="nlpvis-view pa-1 mt-2" height="calc(53% - 48px)" v-intro-step="5" v-intro="'In the information flow, user can analyze how NLP models process a sample through layers in a unified way.'">

              <v-toolbar dense flat>
                <v-btn-toggle dense class="nlpvis-tab"
                  v-model="right_bottom_view"
                >
                  <v-btn value="info_flow" @click="changeRightBottomView()">
                    <span class="nlpvis-title">Information Flow</span>
                  </v-btn>

                  <v-btn value="word_ctx" @click="changeRightBottomView()" v-intro-step="6" v-intro="'In the word context view, user can gain a deep understanding of a word by revealing how the model processes the word based on its context.'">
                    <span class="nlpvis-title">Word Context</span>
                  </v-btn>
                </v-btn-toggle>
                <v-spacer></v-spacer>
                <span :key="`legend${index}`" v-for="(t, index) in labels" class="color-legend" :style="`--color: ${color_scheme[index]}`">
                  {{ t.name }}
                </span>
              </v-toolbar>
              <hr/>
              <svg id="network-svg">
                <!--g id="dag-g"></g-->
              </svg>
            </v-card>
            <v-card class="nlpvis-script pa-1 mt-2" height="calc(7% + 8px)">
              <v-img height="88" width="1340" src="./assets/legend1.png"></v-img>
            </v-card>
          </v-col>
        </v-row>
      </v-container>
    </v-content>
  </v-app>
</template>

<script>
import * as d3 from "d3"
import { mapState, mapMutations, mapActions, mapGetters } from "vuex"
import { hexbin } from "d3-hexbin"
import axios from "axios"
import wordcloud from "./plugins/wordcloud"
import lasso from "./plugins/lasso"
import bus from "./plugins/bus"
import { stepSizeScale, getValuesByEntropy, getWordColor } from "./plugins/utils"
import { paintLineChart } from "./layout/word_contribution_layout"
import { paintWordDAG, paintWordcloud } from "./layout/word_dag_layout"
import WordGroup from "./components/wordgroup"
import InfoTooltip from "./components/infotooltip"

const isalpha = (val) => /^[a-zA-Z]+$/.test(val)

import paintSampleComposition from "./layout/sentence_layout.js";


export default {
  name: "App",

  computed: {
    ...mapState([
      "DAG",
      "linechart",
      "bubble",
      "showed_layers",
      "config",
      "labels",
      "color_scheme",
      "neuron_clusters",
      "all_samples",
      "selected_class",
      "treemap",
      "current_node",
      "current_layers",
      "edges",
      "layers",
      "sentenceclusters",
      "scatterplot",
      "current_viewtype",
      "layout",
      "tooltip",
      "server_url",
      "filter_info",
      "matrix",
    ]),
    ...mapGetters([
      "getSamples",
      "getWordIdxs",
      "view_samples",
      "selected_label",
      "fontSizeRange",
      "getSample",
      "positiveColor",
      "negativeColor",
      "neutralColor",
      "neutralColorGray",
      "getLabelColor",
    ]),
    word_attrs() {
      return {
        frequency: this.frequency_weight / 10,
        uncertainty: this.uncertainty_weight / 10,
        entropy: this.entropy_weight / 10,
      };
    },
  },
  components: {
    WordGroup: WordGroup,
    InfoTooltip: InfoTooltip,
  },
  data: () => ({
    datasets: ['SST-2'],// 'AGNews'],
    models: ['BERT 12-Layer'],// 'Bi-LSTM 4-Layer'],
    chips: [],
    is_lasso_select_all: true,
    enable_word_filter: false,
    left_top_view: "matrix",
    is_wordview_init: false,
    right_bottom_view: "info_flow",
    left_bottom_view: "samples",
    right_top_view: "word",
    enable_word_attr: false,
    enable_tooltip: false,
    show_trending_up: false,
    show_trending_down: false,
    trending_down_lines: [],
    trending_up_lines: [],
    trending_down_underlines: [],
    trending_up_underlines: [],
    enable_word_highlight: false,
    dataset_name: "",
    n_layer: 0,
    sentences: null,
    current_text: "",
    frequency_weight: 10,
    uncertainty_weight: 10,
    entropy_weight: 10,
    infolossWords: [],
    overviewWords: [],
    selectedSentence: null,
    current_items: [],
    current: { items: [], keywords: [] },
    currentGrids: null,
    currentGridsData: null,
    selectedGrids: null,
    lasso: null,
    listOrderKey: "p score",
    listOrder: 1,
    word_cache: {},
    total_accuracy: 100,
    search_expand: false,
    is_refreshing: false,
    lastInfoFlow: null,
    lastWordContext: null,
    labelColor: ["#d7191c", "#1a9641", "#1f77b4", "#ff7f0e"],
  }),
  methods: {
    ...mapMutations([
      "calcDAG",
      "updatePosition",
      "setViewtype",
      "setLayers",
      "setNetwork",
      "setScatterplot",
      "calcEdges",
      "showTooltip",
      "hideTooltip",
      'addHoverTimer',
      'removeHoverTimer',
      'setHoverWord',
    ]),
    ...mapActions([
      "changeCurrentSamples",
      "fetchAllSample",
      "fetchLayerInfo",
      "fetchWordIndex",
      "changeFilteredWord",
    ]),
    async rightClickSearchItem(d) {
      if (isNaN(+d)) {
        await this.fetchWordIndex(d)
        let idxs = this.getWordIdxs(d)
        if (idxs.length >= 8) {
          bus.$emit('add_word_DAG', d)
        }
      } else {
      }
    },
    async changeRightBottomView() {
      if (this.right_bottom_view == 'word_ctx') {
        this.right_bottom_view = 'info_flow'
        await this.addSentenceDAG(this.lastInfoFlow)
      } else {
        this.right_bottom_view = 'word_ctx'
        await this.addWordDAG(this.lastWordContext)
      }
    },
    async clickSearchItem(d) {
      if (isNaN(+d)) {
        await this.fetchWordIndex(d)
        await this.changeFilteredWord(d)
      } else {
        if (+d == 145) d -= 10
        await this.addSentenceDAG(+d)
      }
    },
    removeSearchItem() {
      this.chips.splice(this.chips.indexOf(item), 1)
      this.chips = [...this.chips]
    },
    onClickSearch() {
      if (this.search_expand == false) {
        this.search_expand = true
      } else {
        this.search_expand = false
      }
    },
    wordInteraction(word) {
      let hover_word_item = null;
      let has_request = false;
      let self = this;
      return word
        .on("click", async (d) => {
          let resp = self.word_cache[d.text];
          let idxs = resp.data.idxs;
          if (idxs.length >= 10) self.addWordDAG(d.text);
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
              resp = await axios.post(`${self.server_url}/api/word`, {
                word: d.text,
              });
              self.word_cache[d.text] = resp;
            }
            let idxs = resp.data.idxs;
            let info = resp.data.info;
            for (let i = 1; i < info.length; ++i) {
              info[i] = Math.min(info[i - 1], info[i]);
            }
            self.highlightGrid(idxs);
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
        });
    },
    brushGrid(idxs) {
      if (!idxs || idxs.length == 0) {
        this.currentGrids.style("opacity", 1);
        idxs = [];
      } else {
        idxs = new Set(idxs);
        this.currentGrids.style("opacity", (d) => {
          for (let i = 0; i < d.length; ++i) {
            if (idxs.has(d[i].index)) {
              return 1;
            }
          }
          return 0.4;
        });
      }
      this.changeCurrentSamples([...idxs]);
    },
    getWordFillColor(d) {
      let color = this.getWordColor(d);
      if (color == this.neutralColorGray) return "rgb(172, 196, 220)";
      return color;
    },
    getWordColor(d) {
      return getWordColor(this, d)
    },
    highlightGrid(idxs) {
      if (!idxs || idxs.length == 0) {
        idxs = new Set();
        this.currentGrids
          .select("path.hexagon")
          .style("stroke", "white")
          .style("stroke-width", 0.5)
          .style("opacity", 1);
      } else {
        idxs = new Set(idxs);
        let grids0 = this.currentGrids.filter((d) => {
          for (let i = 0; i < d.length; ++i) {
            if (idxs.has(d[i].index)) {
              return true;
            }
          }
          return false;
        });
        let grids1 = this.currentGrids.filter((d) => {
          for (let i = 0; i < d.length; ++i) {
            if (idxs.has(d[i].index)) {
              return false;
            }
          }
          return true;
        });
        // console.log(grids0.select('path.hexagon'), grids1.select('path.hexagon'))

        grids1
          .select("path.hexagon")
          .style("stroke", "white")
          .style("stroke-width", 0.5)
          .style("opacity", 0.5);

        grids0
          .select("path.hexagon")
          .style("stroke", "orange")
          .style("stroke-width", 1)
          .style("opacity", 1);
      }
    },
    svgResize() {
      let network_rect = document
        .getElementById("network-svg")
        .parentElement.getBoundingClientRect();

      let bubble_rect = document
        .getElementById("overview-svg")
        .parentElement.getBoundingClientRect();

      let linechart_rect = document
        .getElementById("linechart-svg")
        .parentElement.getBoundingClientRect();

      network_rect.width -= 8;
      bubble_rect.width -= 8;
      linechart_rect.width -= 8;
      network_rect.height -= 8;
      bubble_rect.height -= 8 + 30;
      linechart_rect.height -= 8 + 24;

      // console.log(network_rect)

      d3.select("#network-svg")
        .attr("width", network_rect.width)
        .attr("height", network_rect.height);
      /*
        .append("rect")
        .attr("width", network_rect.width)
        .attr("height", network_rect.height)
        .style("stroke", "lightgray")
        .style("stroke-width", 1)
        .style("fill", "none");
        */

      //console.log(rect2)
      d3.select("#overview-svg")
        .attr("width", bubble_rect.width)
        .attr("height", bubble_rect.height);

      d3.select("#linechart-svg")
        .attr("width", linechart_rect.width)
        .attr("height", linechart_rect.height);

      this.updatePosition({
        network: network_rect,
        bubble: bubble_rect,
        linechart: linechart_rect,
      });

      let linechart = d3
        .select("#linechart-g")
        .attr(
          "transform",
          `translate(${this.linechart.x},${this.linechart.y})`
        );

      //let DAG = d3.select('#dag-g')
      //  .attr('transform', `translate(${this.DAG.x},${this.DAG.y})`)
    },
    async paintWordcloud() {
      await paintWordcloud(this)
    },
    async paintMatrix() {
      const self = this;
      const svgDOM = d3.select("#matrix-g").attr("class", "matrixdiagram");

      let margin = { left: 120, top: 150, right: 10, bottom: 10 }
      let padding = 50
      let width = this.bubble.width
      let height = this.bubble.height
      let data = this.matrix;
      const row_sum = data.data.map((d) => d.reduce((a, b) => a + b));
      const ids = data.labels;
      const matrix = ids.map((source, i) =>
        ids.map((target, j) => ({
          source,
          target,
          val: data.data[i][j] / row_sum[i],
        }))
      );

      let color_tp = d3.scaleSequentialSqrt((t) =>
        d3.interpolate("white", "gray")(t)
      );
      let color_fn = d3.scaleSequentialSqrt((t) =>
        d3.interpolate("white", "gray")(t)
      );
      color_fn.domain([
        0,
        d3.max(
          matrix
            .flat()
            .filter((d) => d.source != d.target)
            .map((d) => Number(d.val))
        ),
      ]);
      color_tp.domain([
        0,
        d3.max(
          matrix
            .flat()
            .filter((d) => d.source == d.target)
            .map((d) => Number(d.val))
        ),
      ]);

      let w = width - margin.left - margin.right
      let h = height - margin.top - margin.bottom
      w = h = Math.min(w, h) - padding
      let x = d3
        .scaleBand()
        .rangeRound([0, w])
        .paddingInner(0.0)
        .align(0)
        .domain(ids);

      let svg = svgDOM
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      function colored(d) {
        const val = d.val;
        if (d.source == d.target) return color_tp(+val);
        return color_tp(+val);
        if (+val > 0) return color_fn(+val);
        return "white";
      }

      // text label for the axis
      svg
        .append("text")
        .attr("x", w / 2)
        .attr("y", -41)
        .style("text-anchor", "middle")
        .text("Predicted Class");

      svg
        .append("text")
        .attr("x", -w / 2)
        .attr("y", -100)
        .attr("transform", "rotate(270)")
        .style("text-anchor", "middle")
        .text("Ground Truth");

      let row_line = svg
        .selectAll("g.row")
        .data(matrix)
        .enter()
        .append("g")
        .attr("transform", (d) => `translate(10,${x(d[0].source)})`)
        .each(makeRow);

      let row = row_line
        .append("text")
        .attr("class", "label")
        .attr("x", -4)
        .attr("y", x.bandwidth() / 2)
        .attr("dy", "0.32em")
        .attr("text-anchor", "end")
        .text((d) => d[0].source);

      let column = svg
        .selectAll("g.column")
        .data(matrix)
        .enter()
        .append("text")
        .attr("class", "column label")
        .attr(
          "transform",
          (d) => `translate(${x(d[0].source)},${-x.bandwidth() / 2 - 10})`
        ) //rotate(-90)`)
        .attr("x", 4)
        .attr("y", x.bandwidth() / 2)
        .attr("dy", "0.32em")
        .attr("dx", "0.32em")
        .text((d) => d[0].source);

      function makeRow(rowData) {
        let cell = d3
          .select(this)
          .selectAll(".cell")
          .data(rowData)
          .enter()
          .append("g")
          .attr("class", "cell")
          .attr("transform", (d) => `translate(${x(d.target)}, 0)`);

        cell
          .append("rect")
          .attr("width", x.bandwidth())
          .attr("height", x.bandwidth())
          .style("fill-opacity", 1)
          .style("stroke", "lightgray")
          .style("stroke-width", 0.0)
          .style("fill", (d) => colored(d));

        cell
          .append("text")
          .attr("x", x.bandwidth() / 2)
          .attr("y", x.bandwidth() / 2 + 7)
          .style("text-anchor", "middle")
          .style("font-size", "18px")
          .style("font-weight", 500)
          .style("fill", (d) => (d.val > 0.9 ? "white" : "black"))
          .text((d) => Number(d.val).toFixed(3));

        cell
          .on("mouseover", (d) => {
            d3.selectAll(".cell").style("opacity", "0.1");

            d3.selectAll(".cell")
              .filter(
                (e) =>
                  (e.source == d.source || e.source == d.target) &&
                  (e.target == d.source || e.target == d.target)
              )
              .style("opacity", "1")
              .append("rect")
              .attr("class", "hovered")
              .attr("width", x.bandwidth())
              .attr("height", x.bandwidth())
              .style("fill", "none")
              .style("stroke", "gray")
              .style("stroke-width", 1.5);

            row
              .filter((e) => e[0].source === d.source)
              .style("fill", "#000")
              .style("font-weight", "bold");

            column
              .filter((e) => e[0].source === d.target)
              .style("fill", "#000")
              .style("font-weight", "bold");
          })
          .on("mouseout", () => {
            d3.selectAll(".cell").style("opacity", "1");

            d3.selectAll(".cell").select(".hovered").remove();

            row.style("fill", null).style("font-weight", null);
            column.style("fill", null).style("font-weight", null);
          })
          .on("click", () => {
            setTimeout(() => {
              self.left_top_view = "corpus"
              self.is_wordview_init = 1
            }, 500);
          });
        cell.append("title").text((d) => d.val);
      }
    },
    async paintScatterplot() {
      this.paintMatrix();
      const svg = d3.select("#overview-g");
      let margin_left = this.bubble.left;
      let margin_top = 15;
      let margin_right = this.bubble.top;
      let margin_bottom = 8;
      let width = this.bubble.width - margin_left - margin_right;
      let height = this.bubble.height - margin_top - margin_bottom - 18;
      let data = this.scatterplot;
      let self = this;
      self.dataset_name = this.scatterplot.dataset;
      self.n_layer = this.scatterplot.n_layer;

      const disappear_time = 200;
      const animation_time = 2000;
      const glyph_margin = 10

      svg
        .append("clipPath")
        .attr("id", "clip_path")
        .append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height);

      const main_g = svg
        .append("g")
        .attr("class", "main_layer")
        .attr("transform", `translate(${margin_left}, ${margin_top})`);

      const word_g = svg
        .append("g")
        .attr("class", "word_layer")
        .attr("transform", `translate(${margin_left}, ${margin_top})`);

      const label_g = svg
        .append("g")
        .attr("class", "label_layer")
        .attr("transform", `translate(${margin_left}, ${margin_top})`)
        .style("font-family", "sans-serif")
        .style("font-size", 16);

      label_g
        .append("g")
        .attr("transform", `translate(${3},${-2})`)
        .append("text")
        //.attr('transform', 'rotate(90)')
        .style("font-family", "Roboto, san-serif")
        .text("Prediction Score");

      /*
      label_g
        .append("g")
        .attr("transform", `translate(${width - 6},${-2})`)
        .append("text")
        //.attr('transform', 'rotate(90)')
        .attr("text-anchor", "end")
        .style("font-family", "Roboto, san-serif")
        .text("Accuracy");
        */
      /*
          label_g.append('text')
            .attr('x', width - 6)
            .attr('y', height)
            .attr('text-anchor', 'end')
            .text('Embedding t-SNE')
            */

      let strip_step = 0.0625;
      let xdomain0 = d3.extent(data.scatters, (d) => d.x);
      let xdomain = [0, 1];
      data.scatters.forEach(
        (d) =>
          (d.x =
            ((d.x - xdomain0[0]) / (xdomain0[1] - xdomain0[0])) *
              (xdomain[1] - xdomain[0]) *
              0.97 +
            0.015)
      );
      console.log(xdomain);
      xdomain[0] = Math.floor(xdomain[0] * 20) / 20;
      xdomain[1] = Math.ceil(xdomain[1] * 20) / 20;
      let n_strips = Math.floor((xdomain[1] - xdomain[0]) / strip_step + 1e-7);
      let focus_strip = null;
      let ydomain = d3.extent(data.scatters, (d) => d.y);
      let grid_size = ((width / n_strips / 4) * 0.5) / Math.sqrt(0.75);
      let grid_h = grid_size * Math.sqrt(0.75);

      let ticks = [];
      let strips = [];
      for (let i = 0; i < n_strips; ++i) {
        let left = xdomain[0] + (i * (xdomain[1] - xdomain[0])) / n_strips;
        let right =
          xdomain[0] + ((i + 1) * (xdomain[1] - xdomain[0])) / n_strips;
        ticks.push(left);
        let counts = [0, 0];
        data.scatters
          .filter((d) => d.x >= left && d.x < right)
          .forEach((d) => {
            let label = +(d.label == self.selected_label[0]);
            counts[label] = (counts[label] || 0) + 1;
          });
        let p = 0;
        if (counts[0] + counts[1] > 0) {
          p =
            (left < 0.5 - 1e-7 ? counts[1] : counts[0]) /
            (counts[0] + counts[1]);
        }
        strips.push({
          left,
          right,
          counts,
          weight: 1,
          index: i,
          is_top: left < 0.5 - 1e-7,
          p,
        });
      }
      //console.log('strips', strips)
      ticks.push(xdomain[1]);

      let min_counts = Math.min(
        ...strips.map((d) => Math.min(d.counts[0], d.counts[1]))
      );
      let max_counts = Math.max(
        ...strips.map((d) => Math.max(d.counts[0], d.counts[1]))
      );
      let countScale = d3
        .scaleLog()
        .domain([1, max_counts])
        .range([0.03, 0.35]);

      let get_rangex = (width, strips) => {
        let tot_weight = 0;
        for (let d of strips) tot_weight += d.weight;
        let ret = [1];
        for (let i = 0; i < strips.length; ++i)
          ret.push(ret[i] - strips[i].weight / tot_weight);
        return ret.map((d) => d * width);
      };

      let last_grids = null;
      let last_keywords = [];
      paintMainScatter();

      async function paintMainScatter() {
        let plotview_width = width - (glyph_margin + height / n_strips);
        let rangex = get_rangex(height, strips);
        let rangey = [10, plotview_width];
        let xScale = d3.scaleLinear(ticks, rangex);
        let yScale = d3.scaleLinear(ydomain, rangey);
        let yAxis = d3
          .axisLeft(xScale)
          .tickValues(ticks)
          .tickFormat((d) => Number(d).toFixed(2));
        //let yAxis = d3.axisLeft(yScale)

        let rrange = [(height / n_strips) * 0.15, (height / n_strips) * 0.4];
        let rScale = d3.scaleSqrt(
          d3.extent(strips, (d) => d.counts[0] + d.counts[1]),
          rrange
        );
        let hex = hexbin()
          .radius(grid_size)
          .x((d) => d.x)
          .y((d) => d.y);
        let plots = data.scatters.map((d, index) => ({
          x: yScale(d.y),
          y: xScale(d.x),
          is_correct: d.is_correct,
          pred: d.pred,
          label: d.label,
          index: d.id,
        }));
        let raw_grids = hex(plots);
        // data.keywords.forEach((d) => (d.value = d.weight))
        raw_grids.forEach((d, index) => {
          d.count = 0;
          d.error = 0;
          //d.x += grid_h
          //d.y += grid_h
          if (d.length >= 1) {
            if (d[0].pred == self.selected_label[0]) {
              d.color = self.positiveColor;
              d.error_color = self.negativeColor;
            } else {
              d.color = self.negativeColor;
              d.error_color = self.positiveColor;
            }
          } else {
            d.color = "gray";
          }
          for (let i = 0; i < d.length; ++i) {
            if (d[i].is_correct) {
              d.count += 1;
            } else {
              d.error += 1;
            }
          }
        });
        let currentGrids = raw_grids;
        self.currentGridsData = currentGrids;
        // console.log(currentGrids)
        let gridCountDomain = [
          0,
          1,
          Math.max(...raw_grids.map((d) => d.count)),
        ];
        let gridCountScale = d3.scaleLinear(gridCountDomain, [0, 0.3, 0.8]);
        let gridErrorDomain = [
          0,
          1,
          Math.max(...raw_grids.map((d) => d.error || 0)),
        ];

        let gridErrorScale = d3.scaleLinear(gridErrorDomain, [0, 0.5, 0.8]);
        let total_acc = 0;
        let total_sample = 0;
        strips.forEach((d, i) => {
          d.height = xScale(d.left) - xScale(d.right);
          d.r = rScale(d.counts[0] + d.counts[1]);
          let right_sample = i * 2 + 2 <= n_strips ? d.counts[0] : d.counts[1];
          d.acc = Number(
            (right_sample / (d.counts[0] + d.counts[1])) * 100
          ).toFixed(2);
          total_acc += right_sample;
          total_sample += d.counts[0] + d.counts[1];
          console.log(
            i,
            right_sample,
            d.counts[0] + d.counts[1],
            Number((right_sample / (d.counts[0] + d.counts[1])) * 100).toFixed(
              2
            )
          );
        });
        self.total_accuracy = Number((total_acc / total_sample) * 100).toFixed(
          2
        );

        let arc = d3
          .arc()
          .startAngle((d) => -d.p * Math.PI)
          .endAngle((d) => d.p * Math.PI)
          .innerRadius((d) => 0)
          .outerRadius((d) => rScale(d.counts[0] + d.counts[1]));

        main_g
          .selectAll(".strip")
          .data(strips)
          .join(
            (enter) => {
              let g = enter
                .append("g")
                .attr("class", "strip")
                .attr("transform", (d) => `translate(0, ${xScale(d.right)})`);

              g.append("rect")
                .attr("width", width)
                .attr("height", (d) => d.height)
                .attr("fill", (d, i) =>
                  strips[i].is_top ? self.negativeColor : self.positiveColor
                )
                .style("stroke", "white")
                .style("stroke-width", 1)
                .style("stroke-opacity", (d, i) => {
                  let value =
                    2 *
                    countScale(strips[i].is_top ? d.counts[0] : d.counts[1]);
                  return value > 0.25 ? 0 : value;
                })
                .style("fill-opacity", (d, i) =>
                  countScale(strips[i].is_top ? d.counts[0] : d.counts[1])
                )
                .on("dblclick", (d, i) => {
                  if (focus_strip == null) {
                    d.weight = 5;
                    focus_strip = d;
                  } else if (focus_strip.index == i) {
                    d.weight = 1;
                    focus_strip = null;
                  } else if (focus_strip != null) {
                    focus_strip.weight = 1;
                    d.weight = 5;
                    focus_strip = d;
                  }
                  paintMainScatter();
                });

              let percent_g = g
                .filter((d) => d.counts[0] + d.counts[1] > 0)
                .append("g")
                .attr("class", "percentage")
                .attr(
                  "transform",
                  (d, i) =>
                    `translate(${width - glyph_margin - 8}, ${d.height / 2})`
                );

              percent_g
                .append("circle")
                .attr("r", (d) => d.r)
                .attr("fill", (d, i) =>
                  d.counts[0] >= d.counts[1]
                    ? self.negativeColor
                    : self.positiveColor
                )
                .style("opacity", 1)
                .on("click", (e) => {
                  let idxs = data.scatters
                    .filter(
                      (d) => d.x >= e.left && d.x < e.right && !d.is_correct
                    )
                    .map((d) => d.id);
                  bus.$emit("brush_grid", idxs);
                });

              percent_g
                .append("path")
                .attr("d", arc)
                .attr("fill", (d, i) =>
                  d.counts[0] < d.counts[1]
                    ? self.negativeColor
                    : self.positiveColor
                )
                .style("opacity", 1)
                .on("click", (e) => {
                  let idxs = data.scatters
                    .filter(
                      (d) => d.x >= e.left && d.x < e.right && !d.is_correct
                    )
                    .map((d) => d.id);
                  bus.$emit("brush_grid", idxs);
                });

              /*
              percent_g
                .append("text")
                .text((d, i) => (d.acc >= 99.999 ? "100.0%" : `${d.acc}%`))
                .attr("dx", 15)
                .attr("dy", 5)
                .attr("font-size", "16px");
              */
              return g;
            },
            (update) => {
              update
                .transition()
                .duration(animation_time)
                .attr("transform", (d) => `translate(0, ${xScale(d.right)})`);

              update
                .select("rect")
                .transition()
                .duration(animation_time)
                .attr("height", (d) => d.height);

              update
                .filter((d) => d.counts[0] + d.counts[1] > 0)
                .select("g.percentage")
                .transition()
                .duration(animation_time)
                .attr(
                  "transform",
                  (d, i) =>
                    `translate(${width - glyph_margin - 8}, ${d.height / 2})`
                );
            }
          );

        main_g
          .selectAll(".grid")
          .data(
            currentGrids,
            (d) => Math.floor(d.y) * ~~plotview_width + Math.floor(d.x)
          )
          .join(
            (enter) => {
              let grid = enter
                .append("g")
                .attr("class", "grid")
                .style("opacity", 0);

              grid
                .append("path")
                .attr("class", "hexagon")
                //.attr("clip-path", "url(#clip_path)")
                .attr("d", function (d) {
                  return `M ${d.x} ${d.y} ${hex.hexagon()}`;
                })
                .style("stroke", (d) => (d.count == 0 ? "none" : "white"))
                .style("stroke-width", 1)
                .style("fill", (d) => d.color)
                .style("fill-opacity", (d) => gridCountScale(d.count));

              grid
                .append("path")
                .attr("class", "cross")
                .attr("transform", (d) => `translate(${d.x},${d.y})`)
                .attr("d", `M -3 -3 L 3 3 M 3 -3 L -3 3`)
                .style("stroke", (d) => d.error_color)
                .style("stroke-width", 2.5)
                //.style("opacity", (d) => (d.error ? 0.7 : 0));
                .style("opacity", (d) => gridErrorScale(d.error));

              let hover_grid_item = null;
              grid
                .on("mouseover", function (d) {
                  hover_grid_item = d;
                  d3.select(this)
                    .append("path")
                    .attr("class", "hexagon_outline")
                    .attr("d", function (d) {
                      return `M ${d.x} ${d.y} ${hex.hexagon()}`;
                    })
                    .style("stroke", "orange")
                    .style("stroke-width", 2)
                    .style("fill", "none");
                  /*
                  let text = "";
                  if (d.count && d.error) {
                    text = `${d.count} correct samples and ${d.error} wrong samples`;
                  } else if (d.count) {
                    text = `${d.count} correct samples`;
                  } else {
                    text = `${d.error} wrong samples`;
                  }
                  text = `<p>${text}</p>`;
                  let left = d3.event.pageX + 10;
                  let top = d3.event.pageY - 10;
                  self.showTooltip({ top, left, content: text });

                  const show_detail = async () => {
                    let resp = await self.getSamples(d.map((e) => e.index));
                    text =
                      text +
                      resp
                        .slice(0, 5)
                        .map((e, index) => `#${index + 1}: ${e.text}`)
                        .join("</br>");

                    self.showTooltip({ top, left, content: text });
                  };
                  setTimeout(() => {
                    if (hover_grid_item == d) {
                      show_detail();
                    }
                  }, 500);
                  */
                })
                .on("mouseout", function (d) {
                  hover_grid_item = null;
                  d3.select(this).select("path.hexagon_outline").remove();
                  //self.hideTooltip();
                })
                .on("click", function (d) {
                  //bus.$emit("toggle_grids", d.map((e) => e.index))
                  let idxs = d.map((e) => e.index)
                  let idxset = new Set(idxs)
                  if (idxset.has(145)) {
                    self.addSentenceDAG(145)
                  } else if (idxset.has(1733)) {
                    self.addSentenceDAG(1733)
                  } else {
                    self.addSentenceDAG(idxs[0])
                  }
                  self.refreshSelection(idxs);
                });

              grid.transition().duration(animation_time).style("opacity", 1);
            },
            (update) => {
              update
                .select("path.hexagon")
                .transition()
                .duration(animation_time)
                .attr("d", function (d) {
                  return `M ${d.x} ${d.y} ${hex.hexagon()}`;
                })
                .style("fill", (d) => d.color)
                .style("fill-opacity", (d) => gridCountScale(d.count));

              update
                .select("path.cross")
                .transition()
                .duration(animation_time)
                .attr("transform", (d) => `translate(${d.x},${d.y})`)
                .style("stroke", (d) => d.error_color)
                .style("opacity", (d) => (d.error ? 0.7 : 0));
            },
            (exit) => {
              exit
                .transition()
                .duration(animation_time)
                .style("opacity", 0)
                .remove();
            }
          );

        let grids = main_g.selectAll(".grid");
        self.currentGrids = grids;

        let lasso_start = () => {
          //console.log('start')
        };

        let lasttime = 0;
        let lasso_draw = () => {
          if (new Date() - lasttime <= 100) {
            return;
          }
          lasttime = new Date();
          current_lasso.possibleItems().style("opacity", 1.0);

          if (current_lasso.possibleItems().nodes().length > 0) {
            current_lasso.notPossibleItems().style("opacity", 0.4);
          } else {
            current_lasso.notPossibleItems().style("opacity", 1);
          }
        };

        let lasso_end = () => {
          setTimeout(() => {
            if (self.is_refreshing) return
            if (current_lasso.selectedItems().nodes().length > 0) {
              current_lasso.notSelectedItems().style("stroke", 0.4);
            } else {
              current_lasso.notSelectedItems().style("stroke", 1);
            }
            current_lasso.selectedItems().style("opacity", 1);

            let grids_ = current_lasso.selectedItems().data();
            let idxs = [].concat(...grids_.map((d) => d.map((e) => e.index)));
            console.log("self.is_lasso_select_all", self.is_lasso_select_all);
            if (!self.is_lasso_select_all) {
              idxs = idxs.filter((d) => self.all_samples[d].wrong);
            }
            self.refreshSelection(idxs, grids_);
          }, 100)
        };

        const current_lasso = lasso()
          .closePathDistance(305)
          .closePathSelect(true)
          .targetArea(main_g)
          .items(grids)
          .on("start", lasso_start)
          .on("draw", lasso_draw)
          .on("end", lasso_end);
        self.lasso = current_lasso;

        main_g.select("g.grid-lasso").remove();
        main_g
          .append("g")
          .attr("class", "grid-lasso")
          //.attr('transform', `translate(${0},${0})`)
          .call(current_lasso);

        main_g.select("g.x-axis").remove();

        let axis = main_g
          .append("g")
          .attr("class", "x-axis")
          .attr("transform", `translate(${0},${0})`)
          .call(yAxis);

        axis
          // remove the line between the ticks and the chart
          .select(".domain")
          .remove();

        axis
          .selectAll("text")
          .attr("font-size", "16px")
          //.style('font-weight', 600)
          .style("font-family", "Roboto, san-serif");
        let keywords = data.keywords
        getValuesByEntropy(keywords, self, -1)

        window.keywords = keywords;
        // keywords = keywords.slice(0, 200);
        keywords = JSON.parse(JSON.stringify(keywords));
        let valueRange = d3.extent(keywords, (d) => d.value);

        let nwords = keywords
          .filter((d) => self.getWordColor(d) == self.negativeColor)
          .sort((a, b) => b.value - a.value);
        let pwords = keywords
          .filter((d) => self.getWordColor(d) == self.positiveColor)
          .sort((a, b) => b.value - a.value);

        for (let i = 0; i < pwords.length && i < nwords.length; ++i) {
          if (nwords[i].value < pwords[i].value) {
            nwords[i].value = pwords[i].value - 0.001;
          } else {
            pwords[i].value = nwords[i].value - 0.001;
          }
        }

        keywords = keywords.sort((a, b) => b.value - a.value);

        let sizeScale;

        if (self.n_layer == 4) {
          sizeScale = stepSizeScale(
            self.fontSizeRange,
            keywords.map((d) => d.value),
            4,
            [4, 10, 12]
          );
        } else {
          sizeScale = stepSizeScale(
            self.fontSizeRange,
            keywords.map((d) => d.value),
            4,
            [4, 8, 10]
          );
        }
        keywords.forEach(
          (d) =>
            (d.size = Math.min(sizeScale(d.value), self.fontSizeRange[1]))
        );

        //console.log(JSON.stringify(strwords))
        /*
        if (!last_font_size_distribution) {
          let max_size = global_font_size_range[1];
          last_font_size_distribution = keywords.map(
            (d) => sizeScale(d.value) / max_size
          );
        }
        */
        // console.log(JSON.parse(JSON.stringify(keywords)))

        //word_g.selectAll('.keyword').remove()
        let barriers = [];

        function add_barriers(top, bottom) {
          let filtered_grids = currentGrids.filter(
            (d) => d && d.y >= top && d.y < bottom
          );
          filtered_grids = filtered_grids.sort((a, b) => a.x - b.x);
          if (filtered_grids.length == 0) return;
          let last = filtered_grids[0].x;
          for (let j = 1; j <= filtered_grids.length; ++j) {
            if (
              j != filtered_grids.length &&
              filtered_grids[j].x - filtered_grids[j - 1].x <= grid_size * 30
            ) {
              continue;
            } else {
              let x1 = filtered_grids[j - 1].x + grid_size;
              if (x1 + grid_size * 10 > plotview_width) {
                x1 = plotview_width;
              }
              barriers.push({
                x0: last - grid_size - 5,
                x1: x1 + 5,
                y0: top - 5,
                y1: bottom + 5,
              });
              if (j != filtered_grids.length) {
                last = filtered_grids[j].x;
              }
            }
          }
        }

        for (let i = 0; i < n_strips; ++i) {
          let top = xScale(strips[i].right);
          let bottom = xScale(strips[i].left);
          if (strips[i].weight > 1) {
            let step = (bottom - top) / 3;
            for (let k = 0; k < 3; ++k) {
              add_barriers(top + k * step, top + (k + 1) * step);
            }
          } else {
            add_barriers(top, bottom);
          }
          barriers.push({
            x0: i * 2 <= n_strips ? 0 : plotview_width * 0.66,
            x1: i * 2 <= n_strips ? plotview_width * 0.33 : plotview_width,
            y0: top,
            y1: bottom,
          });
          /*
              barriers.push({
                x0: 0,
                x1: plotview_width,
                y0: bottom - 1,
                y1: bottom + 1,
              })
              */
        }
        barriers.push({
          x0: plotview_width - 40,
          x1: plotview_width,
          y0: height - 15,
          y1: height,
        });
        barriers.push({
          x0: 0,
          x1: plotview_width,
          y0: 0,
          y1: 20,
        });
        barriers.push({
          x0: 0,
          x1: plotview_width,
          y0: height - 20,
          y1: height,
        });
        barriers.push({
          x0: 0,
          x1: 10,
          y0: 0,
          y1: 100,
        });
        // barriers = barriers.filter(d => d.x1 - d.x0 > 15)
        let dict = {};
        last_keywords.forEach((d, i) => {
          dict[d.word] = i;
          d.is_new = false;
        });

        wordcloud()
          .size([plotview_width, height])
          .data(keywords)
          .text((d) => d.word)
          .barriers(barriers)
          .rangex((d) => {
            if (dict[d.word]) {
              let mid = last_keywords[dict[d.word]].x / plotview_width;
              let left = mid - 0.1;
              let right = mid + 0.1;
              return [Math.max(0, left), Math.min(right, 1)];
            } else {
              if (self.getWordColor(d) == self.positiveColor) {
                return [0, 0.5, 0.25];
              } else if (self.getWordColor(d) == self.negativeColor) {
                return [0.6, 1, 0.82];
              } else {
                return [0.3, 0.8, 0.55];
              }
            }
          })
          .padding(6)
          .rangey((d) => {
            if (dict[d.word]) {
              let mid = last_keywords[dict[d.word]].y / height;
              let left = mid - 0.1;
              let right = mid + 0.1;
              return [Math.max(0, left), Math.min(right, 1), mid];
            } else {
              if (self.getWordColor(d) == self.positiveColor) {
                return [0.1, 0.4, 0.25];
              } else if (self.getWordColor(d) == self.negativeColor) {
                return [0.4, 1, 0.65];
              } else {
                return [0.2, 0.6, 0.5];
              }
              let alpha = 1,
                beta = 1;
              if (d.score.mean < 0.47) (alpha = 0.5), (beta = 1.5);
              if (d.score.mean > 0.53) (alpha = 1.5), (beta = 0.5);
              let std = Math.max(0.05, d.score.std);
              let left = xScale(d.score.mean + std * alpha) / height;
              let right = xScale(d.score.mean - std * beta) / height;
              let mid = xScale(d.score.mean) / height;
              if (left < 0.5) mid = left;
              else if (right > 0.5) mid = right;
              return [Math.max(0, left), Math.min(1, right), mid];
            }
          })
          .fontSize((d) => d.size)
          .font("Roboto")
          .fontWeight((d) => (sizeScale(d.value) >= 0 ? "bold" : "normal"))
          .start((words) => {
            // console.log(words)
            let all_words = words
              .filter((d) => d.display)
              .map((d) => ({
                x: d.x,
                y: d.y,
                layer: 0,
                size: d.size,
                text: d.text,
                display: d.display,
                width: d.width,
                height: d.height,
                weight: "bold",
                id: 'corpus',
                highlight: d.highlight,
                fill: self.getWordColor(d),
                contri: d.contri,
                /*
                  d.score.mean < 0.465
                    ? self.negativeColor
                    : d.score.mean > 0.535
                    ? self.positiveColor
                    : "gray",
                    */
              }));
            let nwords = all_words.filter(
              (d) => self.getWordColor(d) == self.negativeColor
            );
            let pwords = all_words.filter(
              (d) => self.getWordColor(d) == self.positiveColor
            );
            let mwords = all_words.filter(
              (d) =>
                self.getWordColor(d) != self.negativeColor &&
                self.getWordColor(d) != self.positiveColor
            );
            self.overviewWords = nwords
              .slice(0, 25)
              .concat(pwords.slice(0, 25))
              .concat(mwords.slice(0, 20));
          });
      }
    },
    async paintLinechart(highlight_word = null) {
      await paintLineChart(this, highlight_word, this.fontSizeRange);
    },
    saveWordAttr() {
      this.enable_word_attr = false;
      this.refreshLinechart();
    },
    async refreshLinechart() {
      await this.fetchLayerInfo({
        idxs: (this.current && this.current.idxs) || null,
        attrs: this.word_attrs,
      })
      this.paintLinechart()
    },
    async refreshSelection(idxs, grids = null) {
      this.is_refreshing = true
      let self = this
      await this.fetchLayerInfo({ idxs, attrs: this.word_attrs })
      this.current = {};
      let resp = await self.getSamples(idxs);
      this.current_items = resp.sort((a, b) => a.score - b.score);
      this.current.grids = grids;
      this.current.idxs = idxs;
      this.paintLinechart();
      this.brushGrid(idxs);
      this.is_refreshing = false
    },
    async addSentenceDAG(index) {
      this.lastInfoFlow = index
      await this.refreshDAG("sentence", index)
      await this.paintLinechart()
    },
    async addWordDAG(word) {
      this.lastWordContext = word
      await this.refreshDAG("word", word)
    },
    async refreshDAG(type, idx) {
      let req = { level: type, layers: this.showed_layers }
      if (type == "sentence") {
        req.idx = idx
      } else {
        req.word = idx
      }
      this.setViewtype(type)
      let resp = await axios.post(`${this.server_url}/api/networks`, req)
      if (type == "sentence") {
        this.current_text = this.getSample(idx).text
        this.setLayers(resp.data.linechart)
      }
      this.setNetwork(resp)
      this.calcDAG()
      this.paintDAG()
    },
    async paintDAG() {
      if (this.current_viewtype == "word") {
        this.right_bottom_view = "word_ctx";
        paintWordDAG(this);
        await this.paintWordcloud();
      } else if (this.current_viewtype == "sentence") {
        this.right_bottom_view = "info_flow";
        await paintSampleComposition(this);
      }
    },
  },
  async mounted() {

    function createPdf(elem_id="body") {
      let options = {
          useCSS: true,
      };
      let area = document.getElementById(elem_id);
      let w = Number(area.getAttribute("width"));
      let h = Number(area.getAttribute("height"));
      // let w = 1920, h = 1080;
      let doc = new PDFDocument({compress: false, size: [w, h]});
      console.log(w, h);
      SVGtoPDF(doc, area, 0, 0, options);
      let stream = doc.pipe(blobStream());
      stream.on('finish', function() {
        let blob = stream.toBlob('application/pdf');
        if (navigator.msSaveOrOpenBlob) {
          navigator.msSaveOrOpenBlob(blob, 'File.pdf');
        } else {
          document.getElementById('pdf-file').contentWindow.location.replace(URL.createObjectURL(blob));
        }
      });
      doc.end();
    }
    window.createPdf = createPdf
    /*
    if (window.innerWidth < 2800) {
      console.log('document.body.style.zoom', window.innerWidth / 2800)
      const ratio = window.innerWidth / 2800 
      document.body.parentElement.style.zoom = ratio
      const height = window.innerHeight / ratio - 50
      document.body.parentElement.style.height = `${height}px`
      document.body.style.height = `${height}px`
      console.log(this.$refs.mainview)
      this.$refs.mainview.$el.style.height =  `${height}px`
      this.$refs.simpletable.$el.style['max-height'] =  `${height * 0.52 - 96}px`
    }
    */
    const self = this
    self.svgResize();
    document.getElementsByClassName("lds-facebook")[0].style.display = "none";
    let resp = await axios.post(`${self.server_url}/api/scatterplot`, {});
    await self.setScatterplot(resp);
    window.onresize = () => {
      // self.svgResize();
    };
    this.$intro().start()
    await self.fetchAllSample();
    await self.paintScatterplot();
    await self.refreshSelection(null);
    bus.$on("add_word_DAG", (word) => self.addWordDAG(word));
    bus.$on("add_sentence_DAG", (idx) => self.addSentenceDAG(idx));
    bus.$on("highlight_grid", (idxs) => self.highlightGrid(idxs));
    bus.$on("brush_grid", (idxs) => self.brushGrid(idxs))
  },
};
</script>

<style>
.lds-facebook {
  display: inline-block;
  position: relative;
  z-index: 0;
  width: 80px;
  height: 80px;
  left: 45vw;
  top: 45%;
}

html, body {
  height: 100%;
  width: 100%;
}

#app {
  z-index: 1;
}

.lds-facebook div {
  display: inline-block;
  position: absolute;
  left: 8px;
  width: 16px;
  background: #555;
  animation: lds-facebook 1.2s cubic-bezier(0, 0.5, 0.5, 1) infinite;
}
.lds-facebook div:nth-child(1) {
  left: 8px;
  animation-delay: -0.24s;
}
.lds-facebook div:nth-child(2) {
  left: 32px;
  animation-delay: -0.12s;
}
.lds-facebook div:nth-child(3) {
  left: 56px;
  animation-delay: 0;
}
@keyframes lds-facebook {
  0% {
    top: 8px;
    height: 64px;
  }
  50%,
  100% {
    top: 24px;
    height: 32px;
  }
}

#network-svg {
  -webkit-transition: opacity 1s ease-in-out;
  -moz-transition: opacity 1s ease-in-out;
  -o-transition: opacity 1s ease-in-out;
  transition: opacity 1s ease-in-out;
}

.core-view {
  height: 100%;
  background-color: #dddddd;
}

.max-height {
  height: 100%;
}

.my-subtitle {
  font-size: 18px;
  color: rgba(0, 0, 0, 0.6);
}

.corpus-view {
  background: #fff;
  box-shadow: 0 2px 4px rgba(26, 26, 26, 0.3);
}

.nlpvis-view {
  background: #fff;
  box-shadow: 0 2px 4px rgba(26, 26, 26, 0.3);
}

.nlpvis-script {
  background: rgba(255, 255, 255, 0.99);
  box-shadow: 0 2px 4px rgba(26, 26, 26, 0.3);
}

.nlpvis-tab {
  border-left: 1px solid rgba(0, 0, 0, 0.4);
  border-top: 1px solid rgba(0, 0, 0, 0.4);
  border-bottom: 1px solid white;
}

.nlpvis-last-tab {
  border-left: 1px solid rgba(0, 0, 0, 0.4);
  border-right: 1px solid rgba(0, 0, 0, 0.4);
  border-top: 1px solid rgba(0, 0, 0, 0.4);
  border-bottom: 1px solid white;
}

.network-view {
  background: #fff;
  box-shadow: 0 2px 4px rgba(26, 26, 26, 0.3);
}

.lasso path {
  stroke: gray;
  stroke-width: 2px;
}

.lasso .drawn {
  fill-opacity: 0.05;
}

.lasso .loop_close {
  fill: none;
  stroke-dasharray: 4, 4;
}

.lasso .origin {
  fill: #3399ff;
  fill-opacity: 0.5;
}

.nlpvis-text {
  font-size: 20px !important;
}

.text-singleline {
  white-space: nowrap;
}
.nlpvis-tab {
  border-radius: 0.5rem 0.5rem 0 0 !important;
  border: 0 solid #111;
}

.nlpvis-title {
  font-size: 20px !important;
  font-weight: 700 !important;
  text-transform: capitalize !important;
  font-family: Roboto, sans-serif !important;
  justify-content: baseline !important;
  letter-spacing: 0em !important;
  color: #444;
}

.nlpvis-title.disabled {
  color: gray;
}

svg {
  -moz-user-select: none;
  -webkit-user-select: none;
  -ms-user-select: none;
  -khtml-user-select: none;
  user-select: none;
}

.color-legend {
  display: inline-flex;
  align-items: center;
  margin-right: 1em;
  font-size: 18px;
}

.color-legend::before {
  content: "";
  width: 18px;
  height: 18px;
  margin-right: 0.5em;
  background: var(--color);
}

.svg-tooltip {
  font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI",
    Helvetica, Roboto, sans-serif, "Apple   Color Emoji", "Segoe UI Emoji",
    "Segoe UI Symbol";
  background: rgba(250, 250, 250, 0.9);
  border-radius: 0.1rem;
  border: 1px solid #111;
  color: #222;
  display: block;
  font-size: 18px;
  padding: 0.2rem 0.4rem;
  position: absolute;
  text-overflow: ellipsis;
  white-space: pre;
  z-index: 300;
  padding: 0.4rem 0.6rem;
  visibility: hidden;
}

svg.matrixdiagram {
  font: 20px sans-serif;
}

svg.matrixdiagram .label {
  fill: #999;
  font-size: 20px;
  text-anchor: end;
}

svg.matrixdiagram .column.label {
  text-anchor: start;
}

svg.matrixdiagram rect {
  fill: #eee;
  /* stroke: #d62333; */
  stroke: #fff;
  stroke-width: 0;
}

svg.matrixdiagram rect:hover {
  stroke-width: 2px;
}
</style>
