<template>
  <g class="word-group">
    <g class="hover-item-parent" v-if="hoverItem">
      <rect 
        :x="hoverItem.x"
        :y="hoverItem.y - hoverItem.size + 2"
        class="word-hover-item"
        key="new rect"
        :height="hoverItem.size"
        :width="hoverItem.width"
        :style = "{
          'fill' : hoverItem.fill,
          'opacity' : 0.3,
          'stroke' : hoverItem.fill,
          'stroke-width' : '1px',
        }"
      >
      </rect>
    </g>
    <g class="hover-item-parent"
        v-for="(item, index) in temp_items"
        :key="`tmp${index}`"
          :style = "{
            'opacity': hover_word && item.text != hover_word ? 0.3 : 1
          }">
        <text
          class="word-hover-item"
          :x="item.x"
          :y="item.y"
          :dx="item.dx"
          :dy="item.dy"
          :font-size="item.size"
          :font-style="item.style"
          :font-weight="item.weight"
          :style = "{
            'fill': item.fill,
            'stroke': item.stroke,
            'stroke-width': item.stroke_width,
          }"
        >
          {{ item.text }}
        </text>
    </g>

    <transition-group mode="out-in"
      name="word-list" tag="g">
      <template v-if="interaction == 'filter'">
        <rect v-for="item in selected_items"
          :x="item.x"
          :y="item.y - item.size + 2"
          class="word-list-item"
          :key="`${item.key}bg`"
          :height="item.size"
          :width="item.width"
          :style = "{
            'fill' : item.fill,
            'fill-opacity' : 0.15,
            'stroke' : item.fill,
            'stroke-width' : '1px',
          }">
        </rect>
      </template>
      <g v-for="item in items"
        class="word-list-item"
          @mouseenter="onMouseOver(item)"
          @mouseout="onMouseOut(item)"
          @click="onClick(item)"
          @contextmenu.prevent="onRightClick(item)"
          :key="item.key"
        >
        <text
          v-if="item.display"
          :x="item.x"
          :y="item.y"
          :dx="item.dx"
          :dy="item.dy"
          :font-size="item.size"
          :font-style="item.style"
          :font-weight="item.weight"
          :class="{
            'highlight':  !highlight || highlight_text.indexOf(item.text) != -1,
          }"
          :style = "{
            'opacity': hover_word && item.text != hover_word ? 0.3 : 1,
            'fill': item.fill,
            'stroke': item.stroke,
            'stroke-width': item.stroke_width,
          }"
        >
          {{ item.text }}
        </text>
      </g>
      <g v-for="item in glyphed_items"
        :key="`${item.key}glyph`"
        :transform="`translate(${item.x - 5 - (hover_word && item.text == hover_word && !item.glyph.show ? 10 : 0)},${item.y - item.size / 2 + 5}) rotate(-90)`">
          <g :style = "{
            'opacity': hover_word && item.text != hover_word ? 0.3 : 1
          }">
            <path :d="item.glyph.neu" :fill="neutralColor" stroke="none"></path>
            <path :d="item.glyph.pos" :fill="positiveColor" stroke="none"></path>
            <path :d="item.glyph.neg" :fill="negativeColor" stroke="none"></path>
          </g>
      </g>
    </transition-group>
  </g>  
</template>

<script>
import { mapGetters, mapMutations, mapActions, mapState } from 'vuex'
import bus from '../plugins/bus'
import * as d3 from 'd3'

const sleep = (timeout) => new Promise((resolve) => {
  setTimeout(resolve, timeout);
})

const arc = (x, y, r, start, end) => {
  let path = d3.path()
  path.arc(x, y, r, start, end)
  return `M 0 0 L ${r * Math.cos(start)} ${r * Math.sin(start)} ${path._} L 0 0`
}

const isalpha = val => /^[a-zA-Z]+$/.test(val)
const islegal = val => isalpha(val) && val.length > 1 && val.indexOf('_') == -1

export default {
  data: function() {
    return {
      hoverItem: null,
      hoverEnterTime: 500,
      glyphHeight: 4,
      glyphRadius: 7.5,
    }
  },
  computed: {
    ...mapState([ 'filtered_words', 'hover_word' ]),
    ...mapGetters(['positiveColor', 'checkWordIndex', 'negativeColor', 'neutralColor','getWordIdxs', 'getSamples' ]),
    items() {
      let cnt = {}
      let ret = this.data.map(d => {
        cnt[d.text] = (cnt[d.text] || 0) + 1
        let tot = d.contri && (d.contri.pos + d.contri.neg + d.contri.neu)
        let arcs = { show: d.glyph || 0 }
        if (d.contri) {
          let pos = d.contri.pos / tot * Math.PI * 2
          let neg = d.contri.neg / tot * Math.PI * 2
          let neu = d.contri.neu / tot * Math.PI * 2
          let pos_arc = arc(0, 0, this.glyphRadius, 0, pos)
          let neg_arc = arc(0, 0, this.glyphRadius, pos, pos + neg)
          let neu_arc = arc(0, 0, this.glyphRadius, 0, Math.PI * 2)
          arcs = { show: d.glyph, pos: pos_arc, neg: neg_arc, neu: neu_arc }
        }
        return {
          x: d.x,
          y: d.y,
          dx: (d.dx || 0) + (d.glyph ? 5 : 0),
          dy: d.dy,
          display: d.display,
          //contri: d.contri && [d.contri.pos / tot, d.contri.neu / tot, d.contri.neg / tot] || [],
          glyph: arcs,
          size: d.size,
          width: d.width,
          height: d.height,
          weight: d.weight,
          style: d.style || 'normal',
          text: d.text,
          key: `${d.text}${d.layer}_${d.id}`,
          fill: d.fill,
          stroke: d.stroke,
          stroke_width: d.stroke_width,
          // hover: false,
        }
      }).filter(d => islegal(d.text))
      return ret
    },
    temp_items() {
      return this.items.filter(item => !item.display && item.text == this.hover_word)
    },
    glyphed_items() {
      return this.items.filter(item => item.glyph.show || item.text == this.hover_word)
    },
    selected_items() {
      const filter_set = new Set(this.filtered_words)
      return this.items.filter(d => filter_set.has(d.text))
    }
  },
  props: {
    data: Array,
    tooltip: {
      type: Boolean,
      default: false,
    },
    interaction: {
      type: String,
      default: 'DAG',
    },
    highlight: {
      type: Boolean,
      default: false,
    },
    highlight_text: {
      type: String,
      default: '',
    },
  },
  methods: {
    ...mapActions([ 'fetchWordIndex', 'changeFilteredWord' ]),
    ...mapMutations([ 'showTooltip', 'hideTooltip', 'addHoverTimer', 'removeHoverTimer', 'setHoverWord' ]),
    async onHoverEvent(d, left, top) {
      this.hoverItem = d
      this.setHoverWord(d.text)
      await this.fetchWordIndex(d.text)
      if (this.hoverItem != d) return
      let idxs = this.getWordIdxs(d.text)
      bus.$emit('highlight_grid', idxs)
      let text = `<p>${d.text}, ${idxs.length} samples </p>`
      let resp = await this.getSamples(idxs)
      if (this.hoverItem != d) return
      try {
				text = text + resp.slice(0, 8).map((e, index) => `#${e.index}: ${e.text}`).join('</br>')
				if (this.tooltip) {
					this.showTooltip({ top, left, content: text })
				}
			} catch (e) {
				return
			}
    },
    async onMouseOver(d) {
      const left = window.event.pageX
      const top = window.event.pageY
      let timer = setTimeout(() => {
        this.onHoverEvent(d, left, top)
      }, this.hoverEnterTime)
      this.addHoverTimer(timer)
    },
    async onLeaveEvent(d) {
      bus.$emit('highlight_grid', null)
      this.setHoverWord(null)
      this.hoverItem = null
      if (this.tooltip) {
        this.hideTooltip()
      }
    },
    async onMouseOut(d) {
      this.removeHoverTimer()
      this.onLeaveEvent(d)
    },
    async onRightClick(d) {
      await this.fetchWordIndex(d.text)
      let idxs = this.getWordIdxs(d.text)
      if (idxs.length >= 8) {
        bus.$emit('add_word_DAG', d.text)
      }
    },
    async onClick(d) {
      await this.fetchWordIndex(d.text)
      await this.changeFilteredWord(d.text)
    },
  }
}
</script>
<style scoped>
.word-hover-item {
  transition: all 1s;
  opacity: 1;
}
.word-list-item {
  opacity: 1;
}
.word-list-item {
  transition: all 1s;
}
.word-list-enter {
  transition-delay: 1s;
  opacity: 0;
}
.word-list-leave-to {
  opacity: 0;
}
</style>
